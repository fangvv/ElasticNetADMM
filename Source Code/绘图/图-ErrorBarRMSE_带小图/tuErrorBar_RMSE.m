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

%% 找到当前路径
saveFile = p1(1:i(end));
%% MSE 图
CentralizedtestMSE_mid = (CentralizedtestMSE + CentralizedtestMSE_high)/2;
volume_std = (CentralizedtestMSE_high - CentralizedtestMSE)/2;
H1 = figure;

set(gcf,'Units','centimeters','Position',[6 6 14.5 12]);
set(gca,'Position',[.15 .15 .8 .75]);
set(get(gca,'XLabel'),'FontSize',16);

plot(xAxis,testMSE,'s--','linewidth',1.5);

set(gca,'XTickMode','manual','XTick',[1 20 40 60 80 100]); 
set(gca,'FontSize',16);
xlabel('{\it{N}}');
ylabel('RMSE');
grid on;set(gca,'GridLineStyle',':','GridColor','k','GridAlpha',1);
axis([-5, 105,0,10]);
hold on;
errorbar(xAxis,CentralizedtestMSE_mid,volume_std,'>:','linewidth',1.5);
legend('分布式算法','本地化算法','location','NorthWest');
set(gcf,'Units','centimeters','Position',[6 6 14.5 12]);
set(gca,'Position',[.15 .15 .8 .75]);
set(get(gca,'XLabel'),'FontSize',16);
%%% 箭头
annotation('textarrow',[.3,.3],[.2,.3],'linewidth',1.5);
%生成子图
axes('Position',[0.23,0.4,0.33,0.3]);   
% %% 子数据集
xa = 1:3;
testMSE0 = testMSE(xa,:);
CentralizedtestMSE_mid0 = CentralizedtestMSE_mid(xa,:);
volume_std0 = volume_std(xa,:);
xAxis0 = [1 20 40];
% 
plot(xAxis0,testMSE0,'s--','linewidth',1.5);
%set(gca,'XTickMode','manual','XTick',[1 20 40],'YTick',[0 0.02 0.04 0.06 0.08 0.1 0.12 0.14 0.16]); 
set(gca,'XTickMode','manual','XTick',[1 20 40],'YTick',[0 0.05 0.10 0.15 0.20 0.25]); 
set(gca,'FontSize',12);
grid on;set(gca,'GridLineStyle',':','GridColor','k','GridAlpha',1);
axis([0,50,0,0.25]);
hold on;
errorbar(xAxis0,CentralizedtestMSE_mid0,volume_std0,'>:','linewidth',1.5);
%% 保存图像
saveObjImg = [saveFile,'\MSEError.pdf'];
saveas(gcf,saveObjImg);
saveObjImg = [saveFile,'\MSEError.eps'];
saveas(gcf,saveObjImg);
saveObjImg = [saveFile,'\MSEError.png'];
saveas(gcf,saveObjImg);