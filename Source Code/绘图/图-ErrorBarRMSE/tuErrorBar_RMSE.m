p1 = mfilename('fullpath');
i = findstr(p1,'\');
pPath = p1(1:i(end));

ADMMFile = 'scaledSKlearnData.tab.txt_RMSE_R2.txt';
cd(pPath);
dataADMM = load(ADMMFile);
xAxis = dataADMM(:,1);

testMSE = dataADMM(:,5);
testR2 = dataADMM(:,6);
testR2Adjusted = dataADMM(:,7);

CentralizedFile = 'RMSE2.txt';
dataCentralized = load(CentralizedFile);
CentralizedtestMSE = dataCentralized(:,5);
CentralizedtestMSE_high = dataCentralized(:,6);
CentralizedtestR2 = dataCentralized(:,7);
CentralizedtestR2_high = dataCentralized(:,8);
CentralizedtestR2Adjusted = dataCentralized(:,9);
CentralizedtestR2Adjusted_high = dataCentralized(:,10);

CentralizedtestR21 = dataCentralized(:,11);
CentralizedtestR21_high = dataCentralized(:,12);
CentralizedtestR2Adjusted1 = dataCentralized(:,13);
CentralizedtestR2Adjusted1_high = dataCentralized(:,14);

%% �ҵ���ǰ·��
saveFile = p1(1:i(end));
%% MSE ͼ
CentralizedtestMSE_mid = (CentralizedtestMSE + CentralizedtestMSE_high)/2;
volume_std = (CentralizedtestMSE_high - CentralizedtestMSE)/2;
H1 = figure;

set(gcf,'Units','centimeters','Position',[6 6 14.5 12]);
set(gca,'Position',[.15 .15 .8 .75]);
set(get(gca,'XLabel'),'FontSize',16);

plot(xAxis,testMSE,'>:','linewidth',1.8);

set(gca,'XTickMode','manual','XTick',[1 15 30 45 60 90]); 
set(gca,'FontSize',16);
xlabel('{\it{N}}');
ylabel('RMSE');
grid on;set(gca,'GridLineStyle',':','GridColor','k','GridAlpha',0.5);
axis([-5, 65,0,800]);
hold on;
errorbar(xAxis,CentralizedtestMSE_mid,volume_std,'o:','linewidth',1.8);
legend('�ֲ�ʽ�㷨','���ػ��㷨','location','NorthWest');
set(gcf,'Units','centimeters','Position',[6 6 14.5 12]);
set(gca,'Position',[.15 .15 .8 .75]);
set(get(gca,'XLabel'),'FontSize',16);
% %% ��ͷ
% annotation('textarrow',[.3,.3],[.2,.3],'linewidth',1.5);
% %% ������ͼ
% axes('Position',[0.23,0.4,0.33,0.3]);   
% %% �����ݼ�
% xa = 1:2;
% testMSE0 = testMSE(xa,:);
% CentralizedtestMSE_mid0 = CentralizedtestMSE_mid(xa,:);
% volume_std0 = volume_std(xa,:);
% xAxis0 = [1 10];
% 
% plot(xAxis0,testMSE0,'s--','linewidth',1.5);
% set(gca,'XTickMode','manual','XTick',[1 10],'YTick',[0 10000 20000 30000]); 
% set(gca,'FontSize',12);
% grid on;set(gca,'GridLineStyle',':','GridColor','k','GridAlpha',1);
% axis([-5, 15,0,30000]);
% hold on;
% errorbar(xAxis0,CentralizedtestMSE_mid0,volume_std0,'>:','linewidth',1.5);
%% ����ͼ��
saveObjImg = [saveFile,'\MSEError.pdf'];
saveas(gcf,saveObjImg);
saveObjImg = [saveFile,'\MSEError.eps'];
saveas(gcf,saveObjImg);
saveObjImg = [saveFile,'\MSEError.png'];
saveas(gcf,saveObjImg);