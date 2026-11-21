[Setup]
AppName=SbobinaFacile AI
AppVersion=1.0
AppPublisher=Team Interno
DefaultDirName={localappdata}\Programs\SbobinaFacile
DefaultGroupName=SbobinaFacile AI
PrivilegesRequired=lowest
OutputDir=installer_output
OutputBaseFilename=Setup_SbobinaFacile_v1.0
Compression=lzma2
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "italian"; MessagesFile: "compiler:Languages\Italian.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "dist\SbobinaFacile\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\SbobinaFacile AI"; Filename: "{app}\SbobinaFacile.exe"
Name: "{autodesktop}\SbobinaFacile AI"; Filename: "{app}\SbobinaFacile.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\SbobinaFacile.exe"; Description: "{cm:LaunchProgram,SbobinaFacile AI}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{localappdata}\SbobinaFacile"
