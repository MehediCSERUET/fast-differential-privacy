'''Train CIFAR10/CIFAR100 with PyTorch.'''
def main(args):
    config=json.load(open(args.deepspeed_config))

    if args.clipping_mode not in ['nonDP','BK-ghost', 'BK-MixGhostClip', 'BK-MixOpt','nonDP-BiTFiT','BiTFiT']:
        print("Mode must be one of 'nonDP','BK-ghost', 'BK-MixGhostClip', 'BK-MixOpt','nonDP-BiTFiT','BiTFiT'")
        return None


    transformation = torchvision.transforms.Compose([
        torchvision.transforms.Resize(args.dimension),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
    ])

    if torch.distributed.get_rank() != 0:
        # might be downloading cifar data, let rank 0 download first
        torch.distributed.barrier()


    # Data
    if args.cifar_data=='CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root='data/', train=True, download=True, transform=transformation)
        testset = torchvision.datasets.CIFAR10(root='data/', train=False, download=True, transform=transformation)
    elif args.cifar_data=='CIFAR100':
        trainset = torchvision.datasets.CIFAR100(root='data/', train=True, download=True, transform=transformation)
        testset = torchvision.datasets.CIFAR100(root='data/', train=False, download=True, transform=transformation)
    else:
        return "Must specify datasets as CIFAR10 or CIFAR100"
         
 
    if torch.distributed.get_rank() == 0:
        # cifar data is downloaded, indicate other ranks can proceed
        torch.distributed.barrier()

    testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, num_workers=2) #must have num_workers!=0!!!!!! https://github.com/microsoft/DeepSpeed/issues/1735#issuecomment-1025073746

    # Model
    print('==> Building and fixing model..', args.model,'. Mode: ', args.clipping_mode)
    # https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L376
    # embed_dim a.k.a. width, mlp_ratio=MLP/embed_dim, depth is number of blocks
    if args.model!='vitANY':
        net = timm.create_model(args.model,pretrained=True,num_classes=int(args.cifar_data[5:]))
    else:
        net = timm.models.vision_transformer.VisionTransformer(embed_dim=768,num_heads=12,depth=12,mlp_ratio=4,num_classes=int(args.cifar_data[5:]))
    
    if 'BiTFiT' in args.clipping_mode: # not needed for DP-BiTFiT but use here for safety
        for name,param in net.named_parameters():
            if '.bias' not in name:
                param.requires_grad_(False)

    criterion = nn.CrossEntropyLoss()      

    if 'nonDP' not in args.clipping_mode:
        PrivacyEngine_Distributed_extending(
            net,
            batch_size=config['train_batch_size'],
            sample_size=len(trainset),
            epochs=args.epochs,
            target_epsilon=args.epsilon,
            num_GPUs=torch.distributed.get_world_size(),
            torch_seed_is_fixed=(args.seed_fixed>=0), # better use False?
            grad_accum_steps=config['train_batch_size']/config['train_micro_batch_size_per_gpu']/torch.distributed.get_world_size(),
        )

    print('Number of total parameters: ', sum([p.numel() for p in net.parameters()]))
    print(f"Number of trainable parameters: {sum([p.numel() for p in net.parameters() if p.requires_grad])}({sum([p.numel() for p in net.parameters() if p.requires_grad])/sum([p.numel() for p in net.parameters()])})")

    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    # Initialize DeepSpeed to use the following features
    # 1) Distributed model
    # 2) Distributed data loader
    # 3) DeepSpeed optimizer
    model_engine, optimizer, trainloader, __ = deepspeed.initialize(args=args, model=net, optimizer=optimizer, model_parameters=net.parameters(), training_data=trainset)

    fp16 = model_engine.fp16_enabled();bf16 = model_engine.bfloat16_enabled();
    print(f'fp16={fp16},bf16={bf16}')


    def train(epoch):

        net.train()
        train_loss = 0
        correct = 0
        total = 0

   
        for batch_idx, data in enumerate(tqdm(trainloader)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, targets = data[0].to(model_engine.local_rank), data[1].to(model_engine.local_rank)
            if fp16:
                inputs = inputs.half()
            if bf16:
                inputs = inputs.bfloat16()
            outputs = model_engine(inputs)

            loss = criterion(outputs, targets)

            model_engine.backward(loss)
            model_engine.step()
                
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('Epoch: ', epoch, len(trainloader), 'Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    def test(epoch):
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(testloader)):
                inputs, targets = data[0].to(model_engine.local_rank), data[1].to(model_engine.local_rank)
                if fp16:
                    inputs = inputs.half()
                if bf16:
                    inputs = inputs.bfloat16()
                outputs = model_engine(inputs)
                #outputs = net(inputs) # https://github.com/microsoft/DeepSpeedExamples/blob/master/cifar/cifar10_deepspeed.py
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            print('Epoch: ', epoch, len(testloader), 'Test Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    for epoch in range(args.epochs):
        train(epoch)
        test(epoch)


if __name__ == '__main__':
    import deepspeed
    import argparse

    parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
    parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
    parser.add_argument('--epochs', default=5, type=int,
                        help='numter of epochs')
    parser.add_argument('--epsilon', default=2, type=float, help='target epsilon')
    parser.add_argument('--clipping_mode', default='BK-MixOpt', type=str)
    parser.add_argument('--model', default='vit_large_patch16_224', type=str)
    parser.add_argument('--cifar_data', type=str, default='CIFAR100')
    parser.add_argument('--dimension', type=int,default=224)
    parser.add_argument('--seed_fixed', type=int,default=3)

    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()
    
    from fastDP import PrivacyEngine_Distributed_extending

    import torch
    import torchvision
    if args.seed_fixed>=0:
        torch.manual_seed(args.seed_fixed) # if use, need change privacy engine's argument
    import torch.nn as nn
    import torch.optim as optim
    import timm
    from tqdm import tqdm
    import warnings; warnings.filterwarnings("ignore")
    
    import json

    import deepspeed
    deepspeed.init_distributed()

    main(args)
