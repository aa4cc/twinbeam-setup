# Flash Linux4Tegra with LI-IMX477 camera support
In order for the Jetson AGX Xavier unit to work properly with the LI-JXAV-MIPI-ADPT-4CAM adapter and the IMX477 camera, the process given in these instlation [instructions](https://www.dropbox.com/s/qgrr0k59jj91ord/IMX477_R32.1_Xavier_NV_Quad_20190614_Driver_Guide.pdf?dl=0) has to be followed. In order to skip later installations of additional packages it is recommended to install also the Jetpack.

# OpenCV
Install OpenCV 4 with CUDA support by running [this](https://raw.githubusercontent.com/AastaNV/JEP/master/script/install_opencv4.1.1_Jetson.sh) script. To make the compilation process faster, you run the compilation on all CPU cores by modifying the line in instalation script
```
make -j3
```
to 
```
make -j8
```

# RT priorities
Add the following line to the file /etc/security/limits.conf:
```
nvidia           -       rtprio          99
```

Copy the script disabling group rt scheduling (make sure the script is executable): sudo cp JetsonCode/disable_group_rt_scheduling.sh /etc/rc.local

# Dialout group
Add the user `nvidia` to the `dialout` user group to allow connecting to `/dev/tty*` without having sudo rights.
```
sudo usermod -aG dialout nvidia
```

# Sokcpp library
The application running on the Jetson uses sockpp library for handling the network communication. Therefore, this library must be installed.  Follow the instruction in [sockpp GH repository](https://github.com/fpagliughi/sockpp).

# Jetson power modes
Set the default mode to the max power mode . Change the line
```
< PM_CONFIG DEFAULT=2 >
```
to 
```
< PM_CONFIG DEFAULT=0 >
```
in
```
sudo nano /etc/nvpmodel.conf
```