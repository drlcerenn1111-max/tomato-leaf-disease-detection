import os
import numpy as np
import pandas as pd
from skimage import io, img_as_float, img_as_ubyte
from skimage.color import rgba2rgb
from scipy.ndimage import generic_filter
from scipy.ndimage import generic_filter
from scipy.ndimage import convolve
from skimage.util import view_as_windows
from scipy.fftpack import dct
from tqdm import tqdm

class FocusMeasure:
    
    @staticmethod
    def get_all_images(path):
        flist = os.listdir(path)
        return [f"{path}/{fname}" for fname in flist]
        
    @staticmethod
    def graylevel_variance_rgb(image):
        """
        Graylevel variance (Krotkov86) for RGB channels
        """
        image = img_as_float(image)
        
        if image.shape[-1] == 4:
            image = rgba2rgb(image)
            
        fm_channels = {}
        for i, color in enumerate(['R', 'G', 'B']):
            channel = image[:, :, i]
            fm_channels[color] = np.std(channel)
            
        fm_channels['mean'] = np.mean(list(fm_channels.values()))
        return fm_channels
    
    @staticmethod
    def graylevel_local_variance_rgb(image, wsize=3):
        """
        Graylevel local variance (Pech2000) for RGB channels
        """
        image = img_as_float(image)
        
        if image.shape[-1] == 4:
            image = rgba2rgb(image)
            
        fm_channels = {}
        for i, color in enumerate(['R', 'G', 'B']):
            channel = image[:, :, i]
            # Hesapla yerel varyansları
            local_var = generic_filter(channel, np.var, size=(wsize, wsize), mode='reflect')
            fm_channels[color] = np.std(local_var)**2
            
        fm_channels['mean'] = np.mean(list(fm_channels.values()))
        return fm_channels
    
    @staticmethod
    def normalized_graylevel_variance_rgb(image):
        """
        Normalized graylevel variance (Santos97) for RGB channels
        """
        image = img_as_float(image)
        
        if image.shape[-1] == 4:
            image = rgba2rgb(image)
            
        fm_channels = {}
        for i, color in enumerate(['R', 'G', 'B']):
            channel = image[:, :, i]
            fm_channels[color] = np.std(channel)**2 / np.mean(channel) if np.mean(channel) > 0 else 0
            
        fm_channels['mean'] = np.mean(list(fm_channels.values()))
        return fm_channels
    
    @staticmethod
    def energy_of_gradient_rgb(image):
        """
        Energy of gradient (Subbarao92a) for RGB channels
        """
        image = img_as_float(image)
        
        if image.shape[-1] == 4:
            image = rgba2rgb(image)
            
        fm_channels = {}
        for i, color in enumerate(['R', 'G', 'B']):
            channel = image[:, :, i]
            Iy = np.zeros_like(channel)
            Ix = np.zeros_like(channel)
            
            Iy[1:, :] = channel[1:, :] - channel[:-1, :]  # Vertical gradient
            Ix[:, 1:] = channel[:, 1:] - channel[:, :-1]  # Horizontal gradient
            
            FM = Ix**2 + Iy**2
            fm_channels[color] = np.mean(FM)
            
        fm_channels['mean'] = np.mean(list(fm_channels.values()))
        return fm_channels
    
    @staticmethod
    def thresholded_gradient_rgb(image, threshold=0):
        """
        Thresholded gradient (Santos97) for RGB channels
        """
        image = img_as_float(image)
        
        if image.shape[-1] == 4:
            image = rgba2rgb(image)
            
        fm_channels = {}
        for i, color in enumerate(['R', 'G', 'B']):
            channel = image[:, :, i]
            Iy = np.zeros_like(channel)
            Ix = np.zeros_like(channel)
            
            Iy[1:, :] = channel[1:, :] - channel[:-1, :]  # Vertical gradient
            Ix[:, 1:] = channel[:, 1:] - channel[:, :-1]  # Horizontal gradient
            
            FM = np.maximum(np.abs(Ix), np.abs(Iy))
            FM[FM < threshold] = 0
            
            # Prevent division by zero
            if np.sum(FM != 0) > 0:
                fm_channels[color] = np.sum(FM) / np.sum(FM != 0)
            else:
                fm_channels[color] = 0
                
        fm_channels['mean'] = np.mean(list(fm_channels.values()))
        return fm_channels
    
    @staticmethod
    def squared_gradient_rgb(image):
        """
        Squared gradient (Eskicioglu95) for RGB channels
        """
        image = img_as_float(image)
        
        if image.shape[-1] == 4:
            image = rgba2rgb(image)
            
        fm_channels = {}
        for i, color in enumerate(['R', 'G', 'B']):
            channel = image[:, :, i]
            Ix = np.zeros_like(channel)
            Ix[:, 1:] = channel[:, 1:] - channel[:, :-1]  # Horizontal gradient
            
            FM = Ix**2
            fm_channels[color] = np.mean(FM)
            
        fm_channels['mean'] = np.mean(list(fm_channels.values()))
        return fm_channels
     
    @staticmethod
    def helmlis_mean_method_rgb(image, wsize=3):
        """
        Helmli's mean method (Helmli2001) for RGB channels
        """
        image = img_as_float(image)
        
        if image.shape[-1] == 4:
            image = rgba2rgb(image)
            
        # Create average filter
        mean_filter = np.ones((wsize, wsize)) / (wsize * wsize)
        
        fm_channels = {}
        for i, color in enumerate(['R', 'G', 'B']):
            channel = image[:, :, i]
            # Apply mean filter
            U = convolve(channel, mean_filter, mode='reflect')
            
            # Calculate ratio
            R1 = np.ones_like(channel)
            nonzero_mask = (channel != 0)
            R1[nonzero_mask] = U[nonzero_mask] / channel[nonzero_mask]
            
            # Apply Helmli's formula
            FM = np.ones_like(channel)
            FM[nonzero_mask] = 1.0 / R1[nonzero_mask]
            
            # Where U > channel, use R1 instead of 1/R1
            index = (U > channel)
            FM[index] = R1[index]
            
            fm_channels[color] = np.mean(FM)
            
        fm_channels['mean'] = np.mean(list(fm_channels.values()))
        return fm_channels
    
    @staticmethod
    def histogram_entropy_rgb(image):
        """
        Histogram entropy (Krotkov86) for RGB channels
        """
        image = img_as_ubyte(img_as_float(image))
        
        if image.shape[-1] == 4:
            image = rgba2rgb(image)
            
        fm_channels = {}
        for i, color in enumerate(['R', 'G', 'B']):
            channel = image[:, :, i]
            
            # Calculate histogram and probabilities
            hist, _ = np.histogram(channel, bins=256, range=(0, 255))
            prob = hist / float(np.sum(hist))
            
            # Calculate entropy (-sum(p*log2(p)))
            # Avoid log(0) by filtering zeros
            entropy = -np.sum(prob[prob > 0] * np.log2(prob[prob > 0]))
            fm_channels[color] = entropy
            
        fm_channels['mean'] = np.mean(list(fm_channels.values()))
        return fm_channels
    
    @staticmethod
    def histogram_range_rgb(image):
        """
        Histogram range (Firestone91) for RGB channels
        """
        image = img_as_float(image)
        
        if image.shape[-1] == 4:
            image = rgba2rgb(image)
            
        fm_channels = {}
        for i, color in enumerate(['R', 'G', 'B']):
            channel = image[:, :, i]
            fm_channels[color] = np.max(channel) - np.min(channel)
            
        fm_channels['mean'] = np.mean(list(fm_channels.values()))
        return fm_channels
    
    @staticmethod
    def energy_of_laplacian_rgb(image):
        """
        Energy of laplacian (Subbarao92a) for RGB channels
        """
        image = img_as_float(image)
        
        if image.shape[-1] == 4:
            image = rgba2rgb(image)
            
        # Define Laplacian filter
        laplacian = np.array([[0, 1, 0],
                              [1, -4, 1],
                              [0, 1, 0]], dtype=np.float32)
            
        fm_channels = {}
        for i, color in enumerate(['R', 'G', 'B']):
            channel = image[:, :, i]
            
            # Apply Laplacian filter
            lap = convolve(channel, laplacian, mode='reflect')
            fm_channels[color] = np.mean(lap**2)
            
        fm_channels['mean'] = np.mean(list(fm_channels.values()))
        return fm_channels
    
    @staticmethod
    def modified_laplacian_rgb(image):
        """
        Modified Laplacian (Nayar89) for RGB channels
        """
        image = img_as_float(image)
        
        if image.shape[-1] == 4:
            image = rgba2rgb(image)
            
        # Define modified Laplacian filter
        M = np.array([-1, 2, -1], dtype=np.float32)
            
        fm_channels = {}
        for i, color in enumerate(['R', 'G', 'B']):
            channel = image[:, :, i]
            
            # Apply horizontal and vertical filters
            Lx = convolve(channel, M.reshape(1, 3), mode='reflect')
            Ly = convolve(channel, M.reshape(3, 1), mode='reflect')
            
            # Sum absolute values
            FM = np.abs(Lx) + np.abs(Ly)
            fm_channels[color] = np.mean(FM)
            
        fm_channels['mean'] = np.mean(list(fm_channels.values()))
        return fm_channels
    
    @staticmethod
    def variance_of_laplacian_rgb(image):
        """
        Variance of laplacian (Pech2000) for RGB channels
        """
        image = img_as_float(image)
        
        if image.shape[-1] == 4:
            image = rgba2rgb(image)
            
        # Define Laplacian filter
        laplacian = np.array([[0, 1, 0],
                              [1, -4, 1],
                              [0, 1, 0]], dtype=np.float32)
            
        fm_channels = {}
        for i, color in enumerate(['R', 'G', 'B']):
            channel = image[:, :, i]
            
            # Apply Laplacian filter
            lap = convolve(channel, laplacian, mode='reflect')
            fm_channels[color] = np.std(lap)**2
            
        fm_channels['mean'] = np.mean(list(fm_channels.values()))
        return fm_channels
    
    @staticmethod
    def diagonal_laplacian_rgb(image):
        """
        Diagonal laplacian (Thelen2009) for RGB channels
        """
        image = img_as_float(image)
        
        if image.shape[-1] == 4:
            image = rgba2rgb(image)
            
        # Define filters
        M1 = np.array([-1, 2, -1], dtype=np.float32)
        M2 = np.array([[0, 0, -1],
                       [0, 2, 0],
                       [-1, 0, 0]], dtype=np.float32) / np.sqrt(2)
        M3 = np.array([[-1, 0, 0],
                       [0, 2, 0],
                       [0, 0, -1]], dtype=np.float32) / np.sqrt(2)
            
        fm_channels = {}
        for i, color in enumerate(['R', 'G', 'B']):
            channel = image[:, :, i]
            
            # Apply filters
            F1 = convolve(channel, M1.reshape(1, 3), mode='reflect')
            F2 = convolve(channel, M2, mode='reflect')
            F3 = convolve(channel, M3, mode='reflect')
            F4 = convolve(channel, M1.reshape(3, 1), mode='reflect')
            
            # Sum absolute values
            FM = np.abs(F1) + np.abs(F2) + np.abs(F3) + np.abs(F4)
            fm_channels[color] = np.mean(FM)
            
        fm_channels['mean'] = np.mean(list(fm_channels.values()))
        return fm_channels
    
    @staticmethod
    def steerable_filters_rgb(image, wsize=11):
        """
        Steerable filters (Minhas2009) for RGB channels
        """
        image = img_as_float(image)
        
        if image.shape[-1] == 4:
            image = rgba2rgb(image)
            
        # Create Gaussian derivative filters
        N = wsize // 2
        sig = N / 2.5
        x, y = np.meshgrid(np.arange(-N, N+1), np.arange(-N, N+1))
        G = np.exp(-(x**2 + y**2) / (2 * sig**2)) / (2 * np.pi * sig)
        
        Gx = -x * G / (sig**2)
        Gx = Gx / np.sum(np.abs(Gx))
        
        Gy = -y * G / (sig**2)
        Gy = Gy / np.sum(np.abs(Gy))
            
        fm_channels = {}
        for i, color in enumerate(['R', 'G', 'B']):
            channel = image[:, :, i]
            
            # Apply filters at different orientations
            R = np.zeros((channel.shape[0], channel.shape[1], 8))
            
            R[:,:,0] = convolve(channel, Gx, mode='reflect')
            R[:,:,1] = convolve(channel, Gy, mode='reflect')
            
            # Filter at 45 degrees
            R[:,:,2] = np.cos(np.deg2rad(45)) * R[:,:,0] + np.sin(np.deg2rad(45)) * R[:,:,1]
            # Filter at 135 degrees
            R[:,:,3] = np.cos(np.deg2rad(135)) * R[:,:,0] + np.sin(np.deg2rad(135)) * R[:,:,1]
            # Filter at 180 degrees
            R[:,:,4] = np.cos(np.deg2rad(180)) * R[:,:,0] + np.sin(np.deg2rad(180)) * R[:,:,1]
            # Filter at 225 degrees
            R[:,:,5] = np.cos(np.deg2rad(225)) * R[:,:,0] + np.sin(np.deg2rad(225)) * R[:,:,1]
            # Filter at 270 degrees
            R[:,:,6] = np.cos(np.deg2rad(270)) * R[:,:,0] + np.sin(np.deg2rad(270)) * R[:,:,1]
            # Filter at 315 degrees
            R[:,:,7] = np.cos(np.deg2rad(315)) * R[:,:,0] + np.sin(np.deg2rad(315)) * R[:,:,1]
            
            # Take maximum response across all orientations
            FM = np.max(R, axis=2)
            fm_channels[color] = np.mean(FM)
            
        fm_channels['mean'] = np.mean(list(fm_channels.values()))
        return fm_channels
    
    @staticmethod
    def spatial_frequency_rgb(image):
        """
        Spatial frequency (Eskicioglu95) for RGB channels
        """
        image = img_as_float(image)
        
        if image.shape[-1] == 4:
            image = rgba2rgb(image)
            
        fm_channels = {}
        for i, color in enumerate(['R', 'G', 'B']):
            channel = image[:, :, i]
            
            Ix = np.zeros_like(channel)
            Iy = np.zeros_like(channel)
            
            Ix[:, 1:] = channel[:, 1:] - channel[:, :-1]  # Horizontal gradient
            Iy[1:, :] = channel[1:, :] - channel[:-1, :]  # Vertical gradient
            
            FM = np.sqrt(Ix**2 + Iy**2)
            fm_channels[color] = np.mean(FM)
            
        fm_channels['mean'] = np.mean(list(fm_channels.values()))
        return fm_channels
    
    @staticmethod
    def tenengrad_rgb(image):
        """
        Tenengrad (Krotkov86) for RGB channels
        """
        image = img_as_float(image)
        
        if image.shape[-1] == 4:
            image = rgba2rgb(image)
            
        # Define Sobel filters
        sobel_x = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float32)
        
        sobel_y = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]], dtype=np.float32)
            
        fm_channels = {}
        for i, color in enumerate(['R', 'G', 'B']):
            channel = image[:, :, i]
            
            # Apply Sobel filters
            Gx = convolve(channel, sobel_x, mode='reflect')
            Gy = convolve(channel, sobel_y, mode='reflect')
            
            # Calculate gradient magnitude squared
            FM = Gx**2 + Gy**2
            fm_channels[color] = np.mean(FM)
            
        fm_channels['mean'] = np.mean(list(fm_channels.values()))
        return fm_channels
    
    @staticmethod
    def tenengrad_variance_rgb(image):
        """
        Tenengrad variance (Pech2000) for RGB channels
        """
        image = img_as_float(image)
        
        if image.shape[-1] == 4:
            image = rgba2rgb(image)
            
        # Define Sobel filters
        sobel_x = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float32)
        
        sobel_y = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]], dtype=np.float32)
            
        fm_channels = {}
        for i, color in enumerate(['R', 'G', 'B']):
            channel = image[:, :, i]
            
            # Apply Sobel filters
            Gx = convolve(channel, sobel_x, mode='reflect')
            Gy = convolve(channel, sobel_y, mode='reflect')
            
            # Calculate gradient magnitude squared
            G = Gx**2 + Gy**2
            fm_channels[color] = np.std(G)**2
            
        fm_channels['mean'] = np.mean(list(fm_channels.values()))
        return fm_channels
    
    @staticmethod
    def vollaths_correlation_rgb(image):
        """
        Vollath's correlation (Santos97) for RGB channels
        """
        image = img_as_float(image)
        
        if image.shape[-1] == 4:
            image = rgba2rgb(image)
            
        fm_channels = {}
        for i, color in enumerate(['R', 'G', 'B']):
            channel = image[:, :, i]
            
            # Compute autocorrelation
            I1 = np.zeros_like(channel)
            I1[1:, :] = channel[:-1, :]  # Shifted down by 1
            
            I2 = np.zeros_like(channel)
            I2[2:, :] = channel[:-2, :]  # Shifted down by 2
            
            # Calculate Vollath's F4 measure
            F4 = np.sum(channel * I1) - np.sum(channel * I2)
            fm_channels[color] = F4 / channel.size
            
        fm_channels['mean'] = np.mean(list(fm_channels.values()))
        return fm_channels
    
    @staticmethod
    def brenner_gradient_rgb(image):
        """
        Brenner gradient (Santos97) for RGB channels
        """
        image = img_as_float(image)
        
        if image.shape[-1] == 4:
            image = rgba2rgb(image)
            
        fm_channels = {}
        for i, color in enumerate(['R', 'G', 'B']):
            channel = image[:, :, i]
            
            # Compute Brenner's gradient (horizontal)
            Ix = np.zeros_like(channel)
            Ix[:, :-2] = (channel[:, 2:] - channel[:, :-2])**2
            
            # Compute Brenner's gradient (vertical)
            Iy = np.zeros_like(channel)
            Iy[:-2, :] = (channel[2:, :] - channel[:-2, :])**2
            
            # Sum horizontal and vertical components
            FM = Ix + Iy
            fm_channels[color] = np.mean(FM)
            
        fm_channels['mean'] = np.mean(list(fm_channels.values()))
        return fm_channels
    
    @staticmethod
    def sum_of_wavelet_coeffs_rgb(image, level=3):
        """
        Sum of wavelet coefficients (Yang2003) for RGB channels
        """
        try:
            import pywt
        except ImportError:
            print("PyWavelets library is required for wavelet metrics. Please install with 'pip install PyWavelets'")
            return {'R': 0, 'G': 0, 'B': 0, 'mean': 0}
        
        image = img_as_float(image)
        
        if image.shape[-1] == 4:
            image = rgba2rgb(image)
            
        fm_channels = {}
        for i, color in enumerate(['R', 'G', 'B']):
            channel = image[:, :, i]
            
    @staticmethod
    def variance_of_wavelet_coeffs_rgb(image, level=3):
        """
        Variance of wavelet coefficients for RGB channels
        """
        try:
            import pywt
        except ImportError:
            print("PyWavelets library is required for wavelet metrics. Please install with 'pip install PyWavelets'")
            return {'R': 0, 'G': 0, 'B': 0, 'mean': 0}
        
        image = img_as_float(image)
        
        if image.shape[-1] == 4:
            image = rgba2rgb(image)
            
        fm_channels = {}
        for i, color in enumerate(['R', 'G', 'B']):
            channel = image[:, :, i]
            
            # Perform wavelet decomposition
            coeffs = pywt.wavedec2(channel, 'db6', level=level)
            
            # Extract detail coefficients
            all_detail_coeffs = []
            for detail_level in range(1, len(coeffs)):
                for detail_coeffs in coeffs[detail_level]:
                    all_detail_coeffs.extend(detail_coeffs.flatten())
            
            # Calculate variance
            fm_channels[color] = np.var(all_detail_coeffs) if all_detail_coeffs else 0
            
        fm_channels['mean'] = np.mean(list(fm_channels.values()))
        return fm_channels
    
if __name__ == '__main__':

    # ----------------------------------------------------------------
    # Etiket şeması (makaledeki hiyerarşik sınıflandırmaya göre)
    #
    # broad_target  → Model 1 (geniş sınıflandırıcı)
    #   0: Healthy
    #   1: Bacterial
    #   2: Fungal/Oomycete
    #   3: Pest
    #   4: Viral
    #
    # fungal_target → Model 2 (sadece Fungal/Oomycete alt sınıfları)
    #   0: Early Blight
    #   1: Late Blight
    #   2: Leaf Mold
    #   3: Septoria Leaf Spot
    #   4: Target Spot
    #  -1: Fungal değil (bu sınıf Model 2'de kullanılmaz)
    # ----------------------------------------------------------------
    CLASS_INFO = {
        'Tomato___healthy':                                {'broad': 0, 'fungal': -1},
        'Tomato___Bacterial_spot':                         {'broad': 1, 'fungal': -1},
        'Tomato___Early_blight':                           {'broad': 2, 'fungal': 0},
        'Tomato___Late_blight':                            {'broad': 2, 'fungal': 1},
        'Tomato___Leaf_Mold':                              {'broad': 2, 'fungal': 2},
        'Tomato___Septoria_leaf_spot':                     {'broad': 2, 'fungal': 3},
        'Tomato___Target_Spot':                            {'broad': 2, 'fungal': 4},
        'Tomato___Spider_mites Two-spotted_spider_mite':   {'broad': 3, 'fungal': -1},
        'Tomato___Tomato_mosaic_virus':                    {'broad': 4, 'fungal': -1},
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus':          {'broad': 4, 'fungal': -1},
    }

    # Sütun isimleri (Target yerine broad_target ve fungal_target)
    columns = [
        'BrennerR', 'BrennerG', 'BrennerB',
        'DlR', 'DlG', 'DlB',
        'EogR', 'EogG', 'EogB',
        'EolR', 'EolG', 'EolB',
        'HmmR', 'HmmG', 'HmmB',
        'HeR', 'HeG', 'HeB',
        'HrR', 'HrG', 'HrB',
        'SfR', 'SfG', 'SfB',
        'TenR', 'TenG', 'TenB',
        'TvR', 'TvG', 'TvB',
        'VcR', 'VcG', 'VcB',
        'GvR', 'GvG', 'GvB',
        'GlvR', 'GlvG', 'GlvB',
        'NgvR', 'NgvG', 'NgvB',
        'TgR', 'TgG', 'TgB',
        'SgR', 'SgG', 'SgB',
        'MlR', 'MlG', 'MlB',
        'VlR', 'VlG', 'VlB',
        'StR', 'StG', 'StB',
        'broad_target',   # Model 1: 0=Healthy, 1=Bacterial, 2=Fungal/Oomycete, 3=Pest, 4=Viral
        'fungal_target',  # Model 2: 0-4=Fungal alt sınıflar, -1=Fungal değil
    ]

    all_vectors = []

    # Tüm sınıfları döngüyle işle
    for class_folder, labels in CLASS_INFO.items():
        folder_path = f"train/{class_folder}"
        files = FocusMeasure.get_all_images(folder_path)
        print(f"\n[{class_folder}] işleniyor... ({len(files)} görüntü)")

        for image_path in tqdm(files):
            try:
                temp = []
                img = io.imread(image_path)

                # 1. Brenner Gradient
                params = FocusMeasure.brenner_gradient_rgb(img)
                temp.extend([params['R'], params['G'], params['B']])

                # 2. Diagonal Laplacian
                params = FocusMeasure.diagonal_laplacian_rgb(img)
                temp.extend([params['R'], params['G'], params['B']])

                # 3. Energy of Gradient
                params = FocusMeasure.energy_of_gradient_rgb(img)
                temp.extend([params['R'], params['G'], params['B']])

                # 4. Energy of Laplacian
                params = FocusMeasure.energy_of_laplacian_rgb(img)
                temp.extend([params['R'], params['G'], params['B']])

                # 5. Helmlis Mean Method
                params = FocusMeasure.helmlis_mean_method_rgb(img)
                temp.extend([params['R'], params['G'], params['B']])

                # 6. Histogram Entropy
                params = FocusMeasure.histogram_entropy_rgb(img)
                temp.extend([params['R'], params['G'], params['B']])

                # 7. Histogram Range
                params = FocusMeasure.histogram_range_rgb(img)
                temp.extend([params['R'], params['G'], params['B']])

                # 8. Spatial Frequency
                params = FocusMeasure.spatial_frequency_rgb(img)
                temp.extend([params['R'], params['G'], params['B']])

                # 9. Tenengrad
                params = FocusMeasure.tenengrad_rgb(img)
                temp.extend([params['R'], params['G'], params['B']])

                # 10. Tenengrad Variance
                params = FocusMeasure.tenengrad_variance_rgb(img)
                temp.extend([params['R'], params['G'], params['B']])

                # 11. Vollaths Correlation
                params = FocusMeasure.vollaths_correlation_rgb(img)
                temp.extend([params['R'], params['G'], params['B']])

                # 12. Graylevel Variance
                params = FocusMeasure.graylevel_variance_rgb(img)
                temp.extend([params['R'], params['G'], params['B']])

                # 13. Graylevel Local Variance
                params = FocusMeasure.graylevel_local_variance_rgb(img, wsize=3)
                temp.extend([params['R'], params['G'], params['B']])

                # 14. Normalized Graylevel Variance
                params = FocusMeasure.normalized_graylevel_variance_rgb(img)
                temp.extend([params['R'], params['G'], params['B']])

                # 15. Thresholded Gradient
                params = FocusMeasure.thresholded_gradient_rgb(img)
                temp.extend([params['R'], params['G'], params['B']])

                # 16. Squared Gradient
                params = FocusMeasure.squared_gradient_rgb(img)
                temp.extend([params['R'], params['G'], params['B']])

                # 17. Modified Laplacian
                params = FocusMeasure.modified_laplacian_rgb(img)
                temp.extend([params['R'], params['G'], params['B']])

                # 18. Variance of Laplacian
                params = FocusMeasure.variance_of_laplacian_rgb(img)
                temp.extend([params['R'], params['G'], params['B']])

                # 19. Steerable Filters
                params = FocusMeasure.steerable_filters_rgb(img, wsize=11)
                temp.extend([params['R'], params['G'], params['B']])

                # Etiketler
                temp.append(labels['broad'])
                temp.append(labels['fungal'])

                all_vectors.append(temp)

            except Exception as e:
                print(f"Hata: {image_path} → {e}")
                continue

    # Tüm veriyi tek CSV'ye kaydet
    df = pd.DataFrame(all_vectors, columns=columns)
    df.to_csv('tomato_all_features.csv', index=False)
    print(f"\nTamamlandı! Toplam {len(df)} satır → tomato_all_features.csv")
    print(df['broad_target'].value_counts().sort_index())