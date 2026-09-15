import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.time import Time
import astropy.units as u
import tempfile
import os

# --- 页面设置 ---
st.set_page_config(page_title="天文图像方向解析", layout="wide")
st.title("FITS 图像北方向与天顶方向解析")

# --- 你的计算逻辑 (基本保持不变) ---
def calculate_North_and_Zenith_direction(fits_file_path, lon=93.8961, lat=38.6067, height=4200):
    location = EarthLocation(lon=lon * u.deg, lat=lat * u.deg, height=height * u.m)
    hdu = fits.open(fits_file_path)[0]
    wcs = WCS(hdu.header)

    x_center, y_center = hdu.header['CRPIX1'], hdu.header['CRPIX2']
    ra_center, dec_center = wcs.pixel_to_world_values(x_center, y_center)

    dec_north = dec_center + 0.01
    x_north, y_north = wcs.world_to_pixel_values(ra_center, dec_north)
    dx_north = x_north - x_center
    dy_north = y_north - y_center
    north_angle_rad = np.arctan2(dy_north, dx_north)
    north_angle_deg = np.degrees(north_angle_rad)

    obs_time = Time(hdu.header['DATE-OBS'], format='isot', scale='utc')
    target = SkyCoord(ra=ra_center * u.deg, dec=dec_center * u.deg, frame='icrs')
    altaz_frame = AltAz(obstime=obs_time, location=location)
    target_altaz = target.transform_to(altaz_frame)
    zenith = SkyCoord(alt=90 * u.deg, az=target_altaz.az, frame=altaz_frame)
    parallactic_angle = target.position_angle(zenith).deg

    cd1_1 = hdu.header['CD1_1']
    cd1_2 = hdu.header['CD1_2']
    cd2_1 = hdu.header['CD2_1']
    cd2_2 = hdu.header['CD2_2']
    det = cd1_1 * cd2_2 - cd1_2 * cd2_1

    if det < 0:
        zenith_angle_img = north_angle_deg + parallactic_angle
    else:
        zenith_angle_img = north_angle_deg - parallactic_angle

    zenith_angle_img = zenith_angle_img % 360
    zenith_angle_rad = np.radians(zenith_angle_img)

    return north_angle_rad, zenith_angle_rad, hdu

# --- 侧边栏：参数输入与文件上传 ---
with st.sidebar:
    st.header("参数设置")
    lon_input = st.number_input("经度 (deg)", value=93.8961, format="%.4f")
    lat_input = st.number_input("纬度 (deg)", value=38.6067, format="%.4f")
    height_input = st.number_input("海拔 (m)", value=4200.0)
    
    uploaded_file = st.file_uploader("上传 FITS 文件", type=['fits', 'fit'])

# --- 主显示区：绘图 ---
if uploaded_file is not None:
    # 临时保存上传的文件供 astropy 读取
    with tempfile.NamedTemporaryFile(delete=False, suffix=".fits") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        st.write("正在解析文件...")
        north_rad, zenith_rad, hdu = calculate_North_and_Zenith_direction(
            tmp_file_path, lon=lon_input, lat=lat_input, height=height_input
        )
        
        image_data = hdu.data
        x_center, y_center = hdu.header['CRPIX1'], hdu.header['CRPIX2']
        vmin, vmax = np.percentile(image_data, (0.1, 99.9))

        # 第一幅图：DS9 视角 (origin='lower')
        # ---------- 从这里开始替换旧的绘图代码 ----------
        
        # 第一幅图：PHD 视角 (原点在左上) - 放在上面
        st.subheader("PHD 视角 (原点在左上)")
        # 将 figsize 从 (6,4) 放大到 (10,6)，dpi 提高到 120 增加清晰度
        fig2, ax2 = plt.subplots(figsize=(10, 6), dpi=120) 
        im2 = ax2.imshow(image_data, cmap='gray', origin='upper', vmin=vmin, vmax=vmax)
        
        ax2.arrow(x_center, y_center, dx_n, dy_n, color='red', width=1.5, head_width=15, head_length=15)
        ax2.text(x_center + dx_n + 10, y_center + dy_n + 10, 'N', color='red', fontsize=14, fontweight='bold')

        ax2.arrow(x_center, y_center, dx_z, dy_z, color='cyan', width=1.5, head_width=15, head_length=15)
        ax2.text(x_center + dx_z + 10, y_center + dy_z + 10, 'Z', color='cyan', fontsize=14, fontweight='bold')
        
        # use_container_width=True 会让图像自动拉伸，撑满网页的显示区域
        st.pyplot(fig2, use_container_width=True) 

        # 加一条水平分割线，让页面更好看
        st.markdown("---") 

        # 第二幅图：DS9 视角 (原点在左下) - 放在下面
        st.subheader("DS9 视角 (原点在左下)")
        fig1, ax1 = plt.subplots(figsize=(10, 6), dpi=120)
        im1 = ax1.imshow(image_data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        
        ax1.arrow(x_center, y_center, dx_n, dy_n, color='red', width=1.5, head_width=15, head_length=15)
        ax1.text(x_center + dx_n + 10, y_center + dy_n + 10, 'N', color='red', fontsize=14, fontweight='bold')

        ax1.arrow(x_center, y_center, dx_z, dy_z, color='cyan', width=1.5, head_width=15, head_length=15)
        ax1.text(x_center + dx_z + 10, y_center + dy_z + 10, 'Z', color='cyan', fontsize=14, fontweight='bold')
        
        st.pyplot(fig1, use_container_width=True)

    except Exception as e:
        st.error(f"解析出错: {e}")
    finally:
        os.remove(tmp_file_path) # 清理临时文件
else:
    st.info("请在左侧侧边栏上传一个 FITS 文件以开始。")
