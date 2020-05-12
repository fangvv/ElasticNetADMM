function [arrayResult]= tu(fileObjectiveName)
p1 = mfilename('fullpath');
i = findstr(p1,'\');
pPath = p1(1:i(end));
pPath = [pPath,  '\'];
maxIter = 200;

file_list = ScanDir(pPath, fileObjectiveName);
len = length(file_list);

arrayResult = zeros(maxIter,len);
arrayResult(:,1) = 1:1:maxIter;
disp(arrayResult(:,1));
for index = 1:len
    name = file_list(index);
    cd(pPath);
    disp(name{1})
    data = load(name{1});
    disp(size(data));
    x = data(:,1);
    y = data(:,2);
    [m,n] = size(x);
    
    arrayResult(1:m,index+1) = y;
    arrayResult(m+1:maxIter,index+1) = y(m);
    
end

end
%% �������ܣ�ָ��·��path������ͼ��·������ɨ�����ļ���
% path�����ҵ�·��
% file_mask����Ҫ���ҵ��ļ����ͣ�����*.jpg
function file_list = ScanDir(path, file_mask)
file_path =  path;  % ͼ���ļ���·��
img_path_list = dir(strcat(file_path, file_mask)); % ��ȡ���ļ���������jpg��ʽ��ͼ��
% disp(img_path_list);
img_num = length(img_path_list);    % ��ȡͼ��������
file_list = cell(img_num, 1);
if img_num > 0 %������������ͼ��
    for j = 1:img_num %��һ��ȡͼ��
        image_name = img_path_list(j).name;% ͼ����
%         fprintf('��ǰ�ҵ�ָ�����ļ� %s\n', strcat(file_path,image_name));% ��ʾɨ�赽��ͼ��·����
        file_list{j} = image_name;
    end
end
end