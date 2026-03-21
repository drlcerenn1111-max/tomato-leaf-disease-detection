import os
import numpy as np
import pandas as pd
from skimage import io, img_as_float, img_as_ubyte
from skimage.color import rgba2rgb
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

            Iy[1:, :] = channel[1:, :] - channel[:-1, :]
            Ix[:, 1:] = channel[:, 1:] - channel[:, :-1]

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

            Iy[1:, :] = channel[1:, :] - channel[:-1, :]
            Ix[:, 1:] = channel[:, 1:] - channel[:, :-1]

            FM = np.maximum(np.abs(Ix), np.abs(Iy))
            FM[FM < threshold] = 0

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
            Ix[:, 1:] = channel[:, 1:] - channel[:, :-1]

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

        mean_filter = np.ones((wsize, wsize)) / (wsize * wsize)

        fm_channels = {}
        for i, color in enumerate(['R', 'G', 'B']):
            channel = image[:, :, i]
            U = convolve(channel, mean_filter, mode='reflect')

            R1 = np.ones_like(channel)
            nonzero_mask = (channel != 0)
            R1[nonzero_mask] = U[nonzero_mask] / channel[nonzero_mask]

            FM = np.ones_like(channel)
            FM[nonzero_mask] = 1.0 / R1[nonzero_mask]

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

            hist, _ = np.histogram(channel, bins=256, range=(0, 255))
            prob = hist / float(np.sum(hist))

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

        laplacian = np.array([[0, 1, 0],
                              [1, -4, 1],
                              [0, 1, 0]], dtype=np.float32)

        fm_channels = {}
        for i, color in enumerate(['R', 'G', 'B']):
            channel = image[:, :, i]

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

        M = np.array([-1, 2, -1], dtype=np.float32)

        fm_channels = {}
        for i, color in enumerate(['R', 'G', 'B']):
            channel = image[:, :, i]

            Lx = convolve(channel, M.reshape(1, 3), mode='reflect')
            Ly = convolve(channel, M.reshape(3, 1), mode='reflect')

            FM = np.abs(Lx) + np.abs(Ly)
            fm_channels[color] = np.mean(FM)

        fm_channels['mean'] = np.mean(list(fm_channels.values()))
        return fm_channels

    @staticmethod
    def variance_of_laplacian_rgb(image):
        """
        Variance of Laplacian (Pech2000) for RGB channels
        """
        image = img_as_float(image)

        if image.shape[-1] == 4:
            image = rgba2rgb(image)

        laplacian = np.array([[0, 1, 0],
                              [1, -4, 1],
                              [0, 1, 0]], dtype=np.float32)

        fm_channels = {}
        for i, color in enumerate(['R', 'G', 'B']):
            channel = image[:, :, i]

            lap = convolve(channel, laplacian, mode='reflect')
            fm_channels[color] = np.std(lap)**2

        fm_channels['mean'] = np.mean(list(fm_channels.values()))
        return fm_channels

    @staticmethod
    def diagonal_laplacian_rgb(image):
        """
        Diagonal Laplacian for RGB channels
        """
        image = img_as_float(image)

        if image.shape[-1] == 4:
            image = rgba2rgb(image)

        diag_laplacian = np.array([[1, 0, 1],
                                   [0, -4, 0],
                                   [1, 0, 1]], dtype=np.float32)

        fm_channels = {}
        for i, color in enumerate(['R', 'G', 'B']):
            channel = image[:, :, i]

            lap = convolve(channel, diag_laplacian, mode='reflect')
            fm_channels[color] = np.mean(lap**2)

        fm_channels['mean'] = np.mean(list(fm_channels.values()))
        return fm_channels

    @staticmethod
    def steerable_filters_rgb(image, wsize=11):
        """
        Steerable filters for RGB channels
        """
        image = img_as_float(image)

        if image.shape[-1] == 4:
            image = rgba2rgb(image)

        fm_channels = {}
        for i, color in enumerate(['R', 'G', 'B']):
            channel = image[:, :, i]

            gx = np.zeros_like(channel)
            gy = np.zeros_like(channel)
            gx[:, 1:] = channel[:, 1:] - channel[:, :-1]
            gy[1:, :] = channel[1:, :] - channel[:-1, :]

            FM = gx**2 + gy**2
            fm_channels[color] = np.mean(FM)

        fm_channels['mean'] = np.mean(list(fm_channels.values()))
        return fm_channels

    @staticmethod
    def spatial_frequency_rgb(image):
        """
        Spatial frequency for RGB channels
        """
        image = img_as_float(image)

        if image.shape[-1] == 4:
            image = rgba2rgb(image)

        fm_channels = {}
        for i, color in enumerate(['R', 'G', 'B']):
            channel = image[:, :, i]

            RF = np.sqrt(np.mean((channel[:, 1:] - channel[:, :-1])**2))
            CF = np.sqrt(np.mean((channel[1:, :] - channel[:-1, :])**2))
            fm_channels[color] = np.sqrt(RF**2 + CF**2)

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

        Kx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float32)
        Ky = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]], dtype=np.float32)

        fm_channels = {}
        for i, color in enumerate(['R', 'G', 'B']):
            channel = image[:, :, i]

            Gx = convolve(channel, Kx, mode='reflect')
            Gy = convolve(channel, Ky, mode='reflect')

            FM = Gx**2 + Gy**2
            fm_channels[color] = np.mean(FM)

        fm_channels['mean'] = np.mean(list(fm_channels.values()))
        return fm_channels

    @staticmethod
    def tenengrad_variance_rgb(image):
        """
        Tenengrad variance for RGB channels
        """
        image = img_as_float(image)

        if image.shape[-1] == 4:
            image = rgba2rgb(image)

        Kx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float32)
        Ky = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]], dtype=np.float32)

        fm_channels = {}
        for i, color in enumerate(['R', 'G', 'B']):
            channel = image[:, :, i]

            Gx = convolve(channel, Kx, mode='reflect')
            Gy = convolve(channel, Ky, mode='reflect')

            FM = Gx**2 + Gy**2
            fm_channels[color] = np.std(FM)**2

        fm_channels['mean'] = np.mean(list(fm_channels.values()))
        return fm_channels

    @staticmethod
    def vollaths_correlation_rgb(image):
        """
        Vollath's correlation (Vollath87) for RGB channels
        """
        image = img_as_float(image)

        if image.shape[-1] == 4:
            image = rgba2rgb(image)

        fm_channels = {}
        for i, color in enumerate(['R', 'G', 'B']):
            channel = image[:, :, i]

            A1 = np.sum(channel[:, :-1] * channel[:, 1:])
            A2 = np.sum(channel[:, :-2] * channel[:, 2:])
            fm_channels[color] = A1 - A2

        fm_channels['mean'] = np.mean(list(fm_channels.values()))
        return fm_channels

    @staticmethod
    def brenner_gradient_rgb(image):
        """
        Brenner gradient for RGB channels
        """
        image = img_as_float(image)

        if image.shape[-1] == 4:
            image = rgba2rgb(image)

        fm_channels = {}
        for i, color in enumerate(['R', 'G', 'B']):
            channel = image[:, :, i]

            diff = channel[:, 2:] - channel[:, :-2]
            FM = diff**2
            fm_channels[color] = np.mean(FM)

        fm_channels['mean'] = np.mean(list(fm_channels.values()))
        return fm_channels

    @staticmethod
    def sum_of_wavelet_coeffs_rgb(image, level=3):
        """
        Sum of wavelet coefficients for RGB channels
        """
        import pywt
        image = img_as_float(image)

        if image.shape[-1] == 4:
            image = rgba2rgb(image)

        fm_channels = {}
        for i, color in enumerate(['R', 'G', 'B']):
            channel = image[:, :, i]
            coeffs = pywt.wavedec2(channel, 'db6', level=level)
            fm = sum(np.sum(np.abs(c)) for c in coeffs[1:])
            fm_channels[color] = fm

        fm_channels['mean'] = np.mean(list(fm_channels.values()))
        return fm_channels

    @staticmethod
    def variance_of_wavelet_coeffs_rgb(image, level=3):
        """
        Variance of wavelet coefficients for RGB channels
        """
        import pywt
        image = img_as_float(image)

        if image.shape[-1] == 4:
            image = rgba2rgb(image)

        fm_channels = {}
        for i, color in enumerate(['R', 'G', 'B']):
            channel = image[:, :, i]
            coeffs = pywt.wavedec2(channel, 'db6', level=level)
            fm = sum(np.var(c) for c in coeffs[1:])
            fm_channels[color] = fm

        fm_channels['mean'] = np.mean(list(fm_channels.values()))
        return fm_channels


# ─────────────────────────────────────────────
#  Sinif etiket haritasi
# ─────────────────────────────────────────────
BASE_DIR = 'tomato_project/train'

CLASS_MAP = {
    f'{BASE_DIR}/Tomato___Bacterial_spot':                        1,
    f'{BASE_DIR}/Tomato___Early_blight':                          2,
    f'{BASE_DIR}/Tomato___healthy':                               0,
    f'{BASE_DIR}/Tomato___Leaf_Mold':                             4,
    f'{BASE_DIR}/Tomato___Septoria_leaf_spot':                    5,
    f'{BASE_DIR}/Tomato___Spider_mites Two-spotted_spider_mite':  6,
    f'{BASE_DIR}/Tomato___Target_Spot':                           7,
    f'{BASE_DIR}/Tomato___Tomato_mosaic_virus':                   8,
    f'{BASE_DIR}/Tomato___Late_blight':                           3,
    f'{BASE_DIR}/Tomato___Tomato_Yellow_Leaf_Curl_Virus':         9,
}

if __name__ == '__main__':

    columns = [
        # 1. Brenner Gradient
        'BrennerR', 'BrennerG', 'BrennerB',
        # 2. Diagonal Laplacian
        'DlR', 'DlG', 'DlB',
        # 3. Energy of Gradient
        'EogR', 'EogG', 'EogB',
        # 4. Energy of Laplacian
        'EolR', 'EolG', 'EolB',
        # 5. Helmlis Mean Method
        'HmmR', 'HmmG', 'HmmB',
        # 6. Histogram Entropy
        'HeR', 'HeG', 'HeB',
        # 7. Histogram Range
        'HrR', 'HrG', 'HrB',
        # 8. Spatial Frequency
        'SfR', 'SfG', 'SfB',
        # 9. Tenengrad
        'TenR', 'TenG', 'TenB',
        # 10. Tenengrad Variance
        'TvR', 'TvG', 'TvB',
        # 11. Vollaths Correlation
        'VcR', 'VcG', 'VcB',
        # 12. Graylevel Variance
        'GvR', 'GvG', 'GvB',
        # 13. Graylevel Local Variance
        'GlvR', 'GlvG', 'GlvB',
        # 14. Normalized Graylevel Variance
        'NgvR', 'NgvG', 'NgvB',
        # 15. Thresholded Gradient
        'TgR', 'TgG', 'TgB',
        # 16. Squared Gradient
        'SgR', 'SgG', 'SgB',
        # 17. Modified Laplacian
        'MlR', 'MlG', 'MlB',
        # 18. Variance of Laplacian
        'VlR', 'VlG', 'VlB',
        # 19. Steerable Filters
        'StR', 'StG', 'StB',
        # Hedef sınıf etiketi
        'Target',
    ]

    all_vectors = []

    # Her sınıf klasörünü sırayla işle
    for class_path, class_label in CLASS_MAP.items():
        print(f"\n[{class_label}] İşleniyor: {class_path}")
        files = FocusMeasure.get_all_images(class_path)

        for image_path in tqdm(files, desc=class_path.split('/')[-1]):
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

            # Sınıf etiketi
            temp.append(class_label)

            all_vectors.append(temp)

    # Tek DataFrame oluştur ve CSV'ye kaydet
    df = pd.DataFrame(all_vectors, columns=columns)
    df.to_csv('tomato_all_features.csv', index=False)
    print(f"\nTamamlandı! Toplam {len(df)} satır → tomato_all_features.csv")
    print(df['Target'].value_counts().sort_index())
