WINDOWS 
 1. Python 3.8.10 https://www.python.org/ftp/python/3.8.10/python-3.8.10-amd64.exe (64 bit)
 2. CUDA support (for 11.8 CUDA version, you can check this by cmd command "nvidia-smi.exe")
    
        nvidia-smi.exe
                                                                        
      ATTENTION! SOME TOOLS REQUIRE AN NVIDIA ACCOUNT TO DOWNLOAD THEM!

      CUDA Toolkit https://developer.download.nvidia.com/compute/cuda/11.8.0/network_installers/cuda_11.8.0_windows_network.exe

      CUDNN https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.7/local_installers/11.x/cudnn-windows-x86_64-8.9.7.29_cuda11-archive.zip/

      NVIDIA Nsight Graphics https://developer.nvidia.com/downloads/assets/tools/secure/nsight-graphics/2024_1_0/windows/nvidia_nsight_graphics_2024.1.0.24079.msi

      NVIDIA Nsight Integration (you must have Visual Studio) https://marketplace.visualstudio.com/items?itemName=NVIDIA.NvNsightToolsVSIntegration

      NVIDIA Nsight Compute https://developer.nvidia.com/downloads/assets/tools/secure/nsight-compute/2024_1_1/nsight-compute-win64-2024.1.1.4-33998838.msi



      Open the root directory of NVIDIA CUDA

        C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8

      Open the CUDNN archive and drop content to the directory of NVIDIA CUDA:

        1.from archive/bin to /v11.8/bin
        2.from archive/include to /v11.8/include
        3.from archive/lib/x64 to /v11.8/lib/x64

      I suggest restarting the computer, but it's not necessary.
    
3. There are two commands to check:

       nvcc --version

   or

       pip install tensorflow-gpu==2.10.0

   If you have done everything correctly, the 1st command will show you the NVIDIA (R) Cuda compiler driver version, and the 2nd command will install the framework for you.
   If the 1st command is "unknown", do it again, you have made a mistake somewhere. Cheer up, CNN works on CPUs too!

4. Install my repository

     Unarichive this

     Open in IDE 

     Choose the Python 3.8 interpreter at the Run/Debug Configuration

     Go to the terminal and execute this command:

       pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu11

     Run "main.py"

     I suggest you check the Settings.md to get results, not a stove from a computer.

     
       
       
   
    
      
