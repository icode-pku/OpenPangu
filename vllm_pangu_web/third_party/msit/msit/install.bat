::  Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
::
::  Licensed under the Apache License, Version 2.0 (the "License");
::  you may not use this file except in compliance with the License.
::  You may obtain a copy of the License at
::
::  http://www.apache.org/licenses/LICENSE-2.0
::
::  Unless required by applicable law or agreed to in writing, software
::  distributed under the License is distributed on an "AS IS" BASIS,
::  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
::  See the License for the specific language governing permissions and
::  limitations under the License.

@ECHO OFF

SET cwd=%cd%
SET CURRENT_DIR=%~dp0
SET arg_force_reinstall=
SET full_install=
SET llvm_path=
SET mingw_w64_path=
SET skip_check_cert=

SET all_component=1
SET select_surgeon=
SET uninstall=
SET all_uninstall=

:loop
IF NOT "%1"=="" (
    IF "%1"=="--surgeon" (
        SET select_surgeon=1
        SET all_component=0
        SHIFT
    ) ELSE IF "%1"=="--full" (
        SET full_install=--full
        SHIFT
    ) ELSE IF "%1"=="-k"  (
        SET skip_check_cert=-k
        SHIFT
    ) ELSE IF "%1"=="--force-reinstall"  (
        SET arg_force_reinstall=--force-reinstall
        SHIFT
    ) ELSE IF "%1"=="--uninstall"  (
        SET uninstall=1
        SHIFT
    ) ELSE IF "%1"=="-y"  (
        SET all_uninstall=-y
        SHIFT
    ) ELSE IF "%1"=="--llvm"  (
        SET llvm_path=--llvm "%~2"
        SHIFT & SHIFT
    ) ELSE IF "%1"=="--mingw"  (
        SET mingw_w64_path=--mingw "%~2"
        SHIFT & SHIFT
    ) ELSE (
        ECHO unknown option: %1, exiting...
		EXIT /B 0
    )
    GOTO :loop
)

SET PYTHON3=

:: first, try get python3 version by `python3 -V`
python3 -V 2>nul ^| findstr /C:"Python 3." >nul && ( SET PYTHON3=python3 )

:: second, try get python3 version by `python -V`, because python3 may be installed as python.exe on windows
python -V 2>nul ^| findstr /C:"Python 3." >nul && ( SET PYTHON3=python )

IF NOT DEFINED PYTHON3 (
    ECHO "Error: python3 is not installed"
    ECHO Installing aborted. && %cwd:~0,2% && cd "%cwd%" && EXIT /B 1
)

FOR /F "delims=" %%I IN ("pip3.exe") DO (
    IF NOT EXIST %%~$PATH:I (
        ECHO "Error: pip3 is not installed"
        ECHO Installing aborted. && %cwd:~0,2% && cd "%cwd%" && EXIT /B 1
    )
)


:: install or uninstall
IF DEFINED uninstall (
    CALL:uninstall_func
) ELSE (
    CALL:install_func
)

:: existing!!!
IF NOT %errorlevel%==0 (
    ECHO Installing aborted. && %cwd:~0,2% && cd "%cwd%" && EXIT /B 1
) ELSE (
    ECHO Installing finished. && %cwd:~0,2% && cd "%cwd%" && EXIT /B 0
)


:: function definitions

:uninstall_func

IF DEFINED select_surgeon (
    pip3 uninstall msit-surgeon %all_uninstall%
) 

IF %all_component%==1 (
    pip3 uninstall msit msit-surgeon %all_uninstall%
)

GOTO:eof


:install_func
:: install msit component
pip3 install "%CURRENT_DIR%/" %arg_force_reinstall%
IF NOT %errorlevel%==0 (
    ECHO pip install msit failed, please check the failure reason.
    EXIT /B 1
)

IF %all_component%==1 (
    SET select_surgeon=1
)

IF DEFINED select_surgeon (
    @REM install surgeon component
    pip3 install "%CURRENT_DIR%/components/debug/surgeon" %arg_force_reinstall%
    IF NOT %errorlevel%==0 (
        ECHO pip install surgeon failed, please check the failure reason.
        EXIT /B 1
    )
)

GOTO:eof
