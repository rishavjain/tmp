:: replicating qsub command on windows
::

@ECHO OFF

ECHO %*

FOR /L %%i IN (1,1,12) DO (
    SHIFT
)

ECHO %1
SET cmd=

:args
IF "%1"=="" ( GOTO end )
SET cmd=%cmd% %1
SHIFT
GOTO args

:end
::SET cmd=%cmd:~+1%
::%cmd%