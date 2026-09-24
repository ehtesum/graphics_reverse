@echo off
set LOGFILE=%~dp0cl_shim.log
echo [CL_INVOCATION] %* >> "%LOGFILE%"
for %%f in (%*) do (
    if exist "%%~f" (
        echo [FILE_CONTENT of %%~f] >> "%LOGFILE%"
        type "%%~f" >> "%LOGFILE%"
        echo. >> "%LOGFILE%"
    )
)
"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64\cl.exe" %*
