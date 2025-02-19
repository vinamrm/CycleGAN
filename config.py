class Config():
    def __init__(self):
        #basic parameters
        self.dataroot = None #path to images
        self.name = "experiment_name" #name of the experiment. It decides where to store samples and models
        self.gpu_ids = 0 #gpu ids= e.g. 0  0,1,2, 0,2. use -1 for CPU
        self.checkpoints_dir = "./checkpoints" #models are saved here                 

        #model params
        self.model= "cycle_gan"                          
        self.input_nc= 5 # no. of input channels                          
        self.output_nc= 5 # no. of output image channels: 3 for RGB and 1 for grayscale
        self.ngf= 64 # no. of gen filters in the last conv layer                            
        self.ndf= 64  # no. of discrim filters in the first conv layer
        self.netD= "basic" #specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator                          
        self.netG= "resnet_9blocks" #specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]                
        self.n_layers_D= 3  #only used if netD==n_layers        
        self.norm= "instance" #instance normalization or batch normalization [instance | batch | none]
        self.init_type= "normal"  #network initialization [normal | xavier | kaiming | orthogonal]           
        self.init_gain= 0.02  #scaling factor for normal, xavier and orthogonal.                        
        self.no_dropout= False #no dropout for the generator

        
        #dataset params
        self.batch_size = 1 #input batch size     
        self.crop_size= 256                           
        self.dataset_mode =  "multispectral" #multispectral for 5 channel images
        self.direction= "AtoB"                          
        self.display_winsize= 256 #display window size for both visdom and HTML                      
        self.load_size= 256  #scale images to this size                        
        self.max_dataset_size= None #inf on default                      
        self.no_flip= False  #if specified, do not flip the images for data augmentation                       
        self.num_threads= 4  # num threads for loading data                      
        self.preprocess= 'none' #scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]               
        self.serial_batches= False #if true, takes images in order to make batches, otherwise takes them randomly                     

        #wand params
        self.use_wandb= False                                         
        self.wandb_project_name= "CycleGAN-and-pix2pix"

        #additional params
        self.epoch= 'latest' #which epoch to load? set to latest to use latest cached model
        self.verbose= False #if specified, print more debugging information    
        self.load_iter= 0 #which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]
        self.suffix= '' #customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}

        #test params
        self.aspect_ratio = 1.0  #aspect ratio of result images                       
        self.results_dir= "./results/" #saves results here
        self.eval= False                         
        self.num_test= 50 #how many test images to run

        # network saving and loading parameters
        self.save_latest_freq=5000 #'frequency of saving the latest results'
        self.save_epoch_freq=5 #'frequency of saving checkpoints at the end of epochs'
        self.save_by_iter = False #'whether saves model by iteration'
        self.continue_train = False #'continue training: load the latest model'
        self.epoch_count=1 #'the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...'
        self.phase='train' #'train, val, test, etc'

        #train params
        self.n_epochs=100 #'number of epochs with the initial learning rate'
        self.n_epochs_decay=100 #'number of epochs to linearly decay learning rate to zero'
        self.beta1=0.5 #'momentum term of adam'
        self.lr=0.0002 #'initial learning rate for adam'
        self.gan_mode='lsgan' #'the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.'
        self.pool_size=50 #'the size of image buffer that stores previously generated images'
        self.lr_policy='linear' #'learning rate policy. [linear | step | plateau | cosine]'
        self.lr_decay_iters=50 #'multiply by a gamma every lr_decay_iters iterations'               
