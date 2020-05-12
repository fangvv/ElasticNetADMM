p1 = mfilename('fullpath');
i = findstr(p1,'\');
pPath = p1(1:i(end));

ADMMFile = 'sto.tab.v2.txt_RMSE_R2.txt';
cd(pPath);
dataADMM = load(ADMMFile);
xAxis = dataADMM(:,1);

testMSE = dataADMM(:,5);
testR2 = dataADMM(:,6);
testR2Adjusted = dataADMM(:,7);

CentralizedFile = 'RMSE.txt';
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
%% R^2  Adjusted ͼ��ѡ��������Сֵ��MSE�±�ͬ��
CentralizedtestR2Adjusted_mid = (CentralizedtestR2Adjusted + CentralizedtestR2Adjusted_high)/2;
volumeR2Adjusted_std = (CentralizedtestR2Adjusted_high - CentralizedtestR2Adjusted)/2;
H1 = figure;
%% ����ͼ��λ�ü���С
set(gcf,'Units','centimeters','Position',[6 6 14.5 12]);
set(gca,'Position',[.15 .15 .8 .75]);
set(get(gca,'XLabel'),'FontSize',16);

%% �ֲ�ʽͼ��
plot(xAxis,testR2Adjusted,'s--','linewidth',1.5);
set(gca,'XTickMode','manual','XTick',[1 20 40 60 80 100]); 
set(gca,'FontSize',16);
xlabel('{\it{N}}');
ylabel('У�������ϵ�� R_a^2ֵ');
grid on;set(gca,'GridLineStyle',':','GridColor','k','GridAlpha',0.5);
axis([-5, 105,-3,1.5]);
hold on;
errorbar(xAxis,CentralizedtestR2Adjusted_mid,volumeR2Adjusted_std,'>:','linewidth',1.5);
legend('�ֲ�ʽ�㷨','���ػ��㷨','location','SouthWest');
set(gcf,'Units','centimeters','Position',[6 6 14.5 12]);
set(gca,'Position',[.15 .15 .8 .75]);
set(get(gca,'XLabel'),'FontSize',16);

% %% ��ͷ
annotation('textarrow',[.5,.5],[.8,.7],'linewidth',1.5);
% %% ������ͼ
axes('Position',[0.25,0.45,0.4,0.23]);   
% %% �����ݼ�
xa = 1:4;
testR2Adjusted0 = testR2Adjusted(xa,:);
CentralizedtestR2Adjusted_mid0 = CentralizedtestR2Adjusted_mid(xa,:);
volume_std0 = volumeR2Adjusted_std(xa,:);
xAxis0 = [1 20 40 60];
% 
plot(xAxis0,testR2Adjusted0,'s--','linewidth',1.5);
set(gca,'XTickMode','manual','XTick',[1 20 40],'YTick',[0.96 0.97 0.98 0.99 1]); 
set(gca,'FontSize',12);
grid on;set(gca,'GridLineStyle',':','GridColor','k','GridAlpha',1);
axis([-5,65,0.96,1]);
hold on;
errorbar(xAxis0,CentralizedtestR2Adjusted_mid0,volume_std0,'>:','linewidth',1.5);
%% ����ͼ��
% saveObjImg = [saveFile,'\R2Adjusted.png'];
% saveas(gcf,saveObjImg);
saveObjImg = [saveFile,'\R2Adjusted.pdf'];
saveas(gcf,saveObjImg);
saveObjImg = [saveFile,'\R2Adjusted.png'];
saveas(gcf,saveObjImg);
