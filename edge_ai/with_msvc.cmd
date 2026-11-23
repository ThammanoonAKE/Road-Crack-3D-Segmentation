@echo off
rem Runs the rest of the command line inside the MSVC 2022 x64 + CUDA build environment, with ninja on PATH,
rem so torch / ModelOpt can JIT-compile their CUDA extensions (ModelOpt's NVFP4 simulation needs one).
rem Visual Studio 2022 is looked for in its usual folders; set VSDEVCMD to a VsDevCmd.bat to choose another.
rem The CUDA toolkit is CUDA_HOME or, when that is not set, the installer's CUDA_PATH (developed with 12.8).
for %%e in (BuildTools Community Professional Enterprise) do for %%r in ("%ProgramFiles(x86)%" "%ProgramFiles%") do if not defined VSDEVCMD if exist "%%~r\Microsoft Visual Studio\2022\%%e\Common7\Tools\VsDevCmd.bat" set "VSDEVCMD=%%~r\Microsoft Visual Studio\2022\%%e\Common7\Tools\VsDevCmd.bat"
if not defined VSDEVCMD echo with_msvc: Visual Studio 2022 with the C++ tools was not found; set VSDEVCMD to its VsDevCmd.bat 1>&2 & exit /b 1
if not defined CUDA_HOME set "CUDA_HOME=%CUDA_PATH%"
if not defined CUDA_HOME echo with_msvc: set CUDA_HOME to the CUDA toolkit folder 1>&2 & exit /b 1
call "%VSDEVCMD%" -arch=x64 -host_arch=x64 >nul
if errorlevel 1 exit /b 1
set "CUDA_PATH=%CUDA_HOME%"
set "DISTUTILS_USE_SDK=1"
set "TORCH_CUDA_ARCH_LIST=12.0"
set "PATH=%PATH%;%~dp0tools"
%*
exit /b %errorlevel%
