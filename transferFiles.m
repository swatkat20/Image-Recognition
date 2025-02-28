% Set up the FTP connection

ftpServer = '<IP of Raspberry PI>';  % Raspberry Pi IP address
ftpUser = '<username>';  % Raspberry Pi username
ftpPass = '<password>';  % Raspberry Pi password

% Log in to the FTP server using the credentials
ftpObj = ftp(ftpServer, ftpUser, ftpPass);


% Specify the local file and the remote destination path
localFile = '<Local file path>';  % Local file to send
remotePath = '<Raspberry pi file path>';  % Destination path on Raspberry Pi
ftpObj.cd(remotePath);

% Upload the file
mput(ftpObj, localFile);

% Close the FTP connection
close(ftpObj);

disp('File transfer via FTP successful.');
