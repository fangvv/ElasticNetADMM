clear;clc;

%% 找到当前路径
p = mfilename('fullpath');
i = findstr(p,'\');
yalmipFile = p(1:(i(end)-1));
    

dataName = 'sto.tab.v2.txt';
dirName = '\sto.tab.v2_patientNo';

rho = 1;        % 参数值
lambda =0.01;
mu=0.01;
patientNo=20;
while(patientNo<=20)
    [w,v,history] = ElasticConjGradTest(rho, lambda, mu, yalmipFile,patientNo,dataName,dirName);    
    patientNo= patientNo + 5;
end;


% system('shutdown -s');

%  50   :  5.913099  0.763334  -0.754497  39.684914  0.065733  -228.887446
%  1    :  5.888013  0.830030  -0.737326  42.140901  0.210671  -259.392100
