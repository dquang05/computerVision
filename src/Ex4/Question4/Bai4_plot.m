clc; clear; close all;

% 1. Khai báo tên file dữ liệu được xuất từ Python
filename = 'data.txt';

% Kiểm tra file có tồn tại hay không
if ~isfile(filename)
    error('Không tìm thấy file %s. Vui lòng chạy code Python trước để tạo dữ liệu.', filename);
end

% 2. Đọc dữ liệu từ file TXT
% Hàm readtable tự động nhận diện header: Frame, BoxID, Translation, Roll, Pitch, Yaw
data = readtable(filename);

% Phân loại dữ liệu theo BoxID (0: Ô bên trái, 1: Ô bên phải)
box0_data = data(data.BoxID == 0, :);
box1_data = data(data.BoxID == 1, :);

% ==========================================
% 3. VẼ ĐỒ THỊ CHO BOX 0 (Ô vuông thứ nhất)
% ==========================================
if ~isempty(box0_data)
    figure('Name', 'Tracking Data - Box 0', 'Position', [100, 100, 700, 500]);
    
    % Đồ thị độ dịch chuyển (Translation)
    subplot(2, 1, 1);
    plot(box0_data.Frame, box0_data.Translation, 'b-', 'LineWidth', 1.5);
    title('Box 0: Độ dịch chuyển tâm (Translation) giữa các frame');
    xlabel('Khung hình (Frame)');
    ylabel('Dịch chuyển (Pixel)');
    grid on;

    % Đồ thị góc quay (Roll, Pitch, Yaw)
    subplot(2, 1, 2);
    plot(box0_data.Frame, box0_data.Roll, 'r-', 'LineWidth', 1.5); hold on;
    plot(box0_data.Frame, box0_data.Pitch, 'g-', 'LineWidth', 1.5);
    plot(box0_data.Frame, box0_data.Yaw, 'k-', 'LineWidth', 1.5);
    title('Box 0: Góc quay định hướng (Roll, Pitch, Yaw)');
    xlabel('Khung hình (Frame)');
    ylabel('Góc (Độ)');
    legend('Roll', 'Pitch', 'Yaw', 'Location', 'best');
    grid on;
end

% ==========================================
% 4. VẼ ĐỒ THỊ CHO BOX 1 (Ô vuông thứ hai)
% ==========================================
if ~isempty(box1_data)
    figure('Name', 'Tracking Data - Box 1', 'Position', [820, 100, 700, 500]);
    
    % Đồ thị độ dịch chuyển (Translation)
    subplot(2, 1, 1);
    plot(box1_data.Frame, box1_data.Translation, 'b-', 'LineWidth', 1.5);
    title('Box 1: Độ dịch chuyển tâm (Translation) giữa các frame');
    xlabel('Khung hình (Frame)');
    ylabel('Dịch chuyển (Pixel)');
    grid on;

    % Đồ thị góc quay (Roll, Pitch, Yaw)
    subplot(2, 1, 2);
    plot(box1_data.Frame, box1_data.Roll, 'r-', 'LineWidth', 1.5); hold on;
    plot(box1_data.Frame, box1_data.Pitch, 'g-', 'LineWidth', 1.5);
    plot(box1_data.Frame, box1_data.Yaw, 'k-', 'LineWidth', 1.5);
    title('Box 1: Góc quay định hướng (Roll, Pitch, Yaw)');
    xlabel('Khung hình (Frame)');
    ylabel('Góc (Độ)');
    legend('Roll', 'Pitch', 'Yaw', 'Location', 'best');
    grid on;
end