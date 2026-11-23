@echo off
rem Runs the rest of the command line inside the MSVC 2022 x64 + CUDA 12.8 build environment, with ninja on PATH,
rem so torch / ModelOpt can JIT-compile their CUDA extensions (ModelOpt's NVFP4 simulation needs one).
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul
if errorlevel 1 exit /b 1
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA_PATH=%CUDA_HOME%"
set "DISTUTILS_USE_SDK=1"
set "TORCH_CUDA_ARCH_LIST=12.0"
set "PATH=%PATH%;%~dp0tools"
%*
exit /b %errorlevel%
