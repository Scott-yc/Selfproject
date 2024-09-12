public class Main {
    public static void main(String[] args) throws InterruptedException {
        // 初始化 RealSpeed 对象，设置初始速度为 0，Z 轴角速度为 0.02，俯仰角度为 10 度
        RealSpeed speedCalculator = new RealSpeed(0, 0, 0.02, Math.toRadians(10));

        // 模拟传感器输入的加速度数据
        double sensorSpeedX = 0.1;  // X 轴加速度
        double sensorSpeedY = 0.2;  // Y 轴加速度

        // 模拟进行 5 次速度更新，每次间隔 0.1 秒
        for (int i = 0; i < 5; i++) {
            // 更新速度并获取合成速度
            double totalSpeed = speedCalculator.updateSpeed(sensorSpeedX, sensorSpeedY);

            // 打印当前合成速度
            System.out.printf("第 %d 次合成速度: %.4f m/s\n", i + 1, totalSpeed);

            // 模拟每次更新间隔 0.1 秒
            Thread.sleep(100);  // 等待 100 毫秒（0.1 秒）
        }

        // 可以模拟传感器数据变化，例如增加 X 轴和 Y 轴的加速度
        sensorSpeedX = 0.15;
        sensorSpeedY = 0.25;

        // 再次模拟更新速度
        for (int i = 0; i < 5; i++) {
            // 更新速度并获取新的合成速度
            double totalSpeed = speedCalculator.updateSpeed(sensorSpeedX, sensorSpeedY);

            // 打印更新后的合成速度
            System.out.printf("第 %d 次更新后的合成速度: %.4f m/s\n", i + 6, totalSpeed);

            // 模拟每次更新间隔 0.1 秒
            Thread.sleep(100);  // 等待 100 毫秒（0.1 秒）
        }
    }
}
