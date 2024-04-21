statsToggle = True  # log toggle

statsWin = True  # .cmd file log (for windows only), if you want teach at the macOS:

    statsWin = False

Log consists of:

  Date & Time

  Device name (GPU or CPU)

  Quantity of epochs

  Learning rate

  Batch size

  Network accuracy

  Training time

Some ints:

POPITKA_NE_PITKA # quantity of epochs 
LR = 0.00118 # learning rate
MOM = 0.99 # momentum ( out service by AdamW optimizer)
bSize = 64 # batch size 

in datasets you have int: num_workers=0, if you have MacOS num_workers must be more than 0

I have Intel I7-11700 and NVIDIA RTX 3070, my settings is:

    statsToggle = True  
    statsWin = False  
    POPITKA_NE_PITKA = 32 
    LR = 0.00118 
    MOM = 0.99 # momentum ( out service by AdamW optimizer)
    bSize = 64 

76.92% accuracy by 585 sec. The power of the dark side.

macOS i tested 1 time with the settings:

    statsToggle = True  
    statsWin = False  
    POPITKA_NE_PITKA = 4 
    LR = 0.0012 
    MOM = 0.99 # momentum ( out service by AdamW optimizer)
    bSize = 64 

and

    num_workers=2

60% accuracy, total 4 epochs by 4607 sec, inefficiently. 
