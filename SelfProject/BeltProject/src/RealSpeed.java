public class RealSpeed {
    private double previousTime;
    private double initialSpeedX;
    private double initialSpeedY;
    // Z方向的角速度
    private double angleZ;
    // 俯仰角度
    private double pitch;

    // 构造器
    public RealSpeed(double initialSpeedX, double initialSpeedY, double angleZ, double pitch) {
        this.previousTime = System.currentTimeMillis() / 1000.0;  // 获取当前时间，以秒为单位
        this.initialSpeedX = initialSpeedX;
        this.initialSpeedY = initialSpeedY;
        this.angleZ = angleZ;
        this.pitch = pitch;
    }

    // 计算实时速度, 先获取实时时间算出时间差
    public double updateSpeed(double sensorSpeedX, double sensorSpeedY) {
        double currentTime = System.currentTimeMillis() / 1000.0;
        double gapTime = currentTime - previousTime;

        // 使用俯仰角度修正加速度,因为腰带不可能保证绝对平行地面，会成俯仰角度, 将其转化到水平面更加精确
        double correctedAccX = sensorSpeedX * Math.cos(pitch);
        double correctedAccY = sensorSpeedY * Math.cos(pitch);

        // 更新 X 和 Y 方向的速度，通过加速度和时间差进行积分, 通过主程序里面调整积分大小的范围程度便可.
        double updatedSpeedX = initialSpeedX + (correctedAccX * gapTime);
        double updatedSpeedY = initialSpeedY + (correctedAccY * gapTime);

        // 转向带来的影响，根据 Z 轴角速度调整 X 和 Y 方向的速度运用旋转矩阵
        double angleChange = angleZ * gapTime;  // Z 轴的旋转角度变化
        double adjustedSpeedX = updatedSpeedX * Math.cos(angleChange) - updatedSpeedY * Math.sin(angleChange);
        double adjustedSpeedY = updatedSpeedX * Math.sin(angleChange) + updatedSpeedY * Math.cos(angleChange);

        // 计算合成速度更能反应场景需求，因为盲人在位移的时候会发生一定程度上的移动偏移，y轴上的速度必须考虑
        double totalSpeed = Math.sqrt(adjustedSpeedX * adjustedSpeedX + adjustedSpeedY * adjustedSpeedY);

        // 更新上次时间和速度,以便后面的移动，作为后面移动的基础
        previousTime = currentTime;
        initialSpeedX = adjustedSpeedX;
        initialSpeedY = adjustedSpeedY;

        // 返回更新后的合成速度
        return totalSpeed;
    }
}
