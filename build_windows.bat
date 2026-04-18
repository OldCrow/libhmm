@echo off
REM VS generator handles MSVC toolchain internally; vcvars64 is not needed here.
cmake -S "%~dp0" -B "%~dp0build" -G "Visual Studio 17 2022" -A x64 -DBUILD_SHARED_LIBS=OFF -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
cmake --build "%~dp0build" --config Release --parallel
