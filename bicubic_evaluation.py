# bicubic_evaluation.py
import os
import sys
import yaml
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import gc


# Import the temperature preprocessor from the neural network project
from data_preprocessing import TemperatureDataPreprocessor

# Import utilities (assuming you'll copy them from the current project)
from utils.util_calculate_psnr_ssim import calculate_psnr, calculate_ssim


class BicubicEvaluator:
    """Evaluate bicubic interpolation on temperature data"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Setup logging
        self.setup_logging()

        # **CRITICAL CHANGE**: Use the SAME preprocessor as neural network
        self.preprocessor = TemperatureDataPreprocessor(
            target_height=2048,
            target_width=208
        )

        # Results storage
        self.results = {
            'psnr_values': [],
            'ssim_values': [],
            'avg_psnr': 0,
            'avg_ssim': 0,
            'std_psnr': 0,
            'std_ssim': 0
        }

    def setup_logging(self):
        """Setup logging"""
        os.makedirs('results', exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'results/bicubic_eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_temperature_data(self, npz_file, max_samples=1000):
        """Load and preprocess temperature data from NPZ file"""
        self.logger.info(f"Loading data from {npz_file}...")

        data = np.load(npz_file, allow_pickle=True)

        # Get swaths data
        if 'swaths' in data:
            swaths = data['swaths']
        elif 'swath_array' in data:
            swaths = data['swath_array']
        else:
            raise KeyError(f"Neither 'swaths' nor 'swath_array' found in {npz_file}")

        temperatures = []
        metadata = []

        n_samples = min(len(swaths), max_samples)
        self.logger.info(f"Processing {n_samples} samples...")

        for i in tqdm(range(n_samples), desc="Loading temperature data"):
            swath = swaths[i]
            temp = swath['temperature'].astype(np.float32)


            # **FIXED**: Use SAME preprocessing as neural network

            # 1. Crop/pad to target size (same as neural network)
            temp = self.preprocessor.crop_or_pad(temp)

            # 2. Use SAME normalization as neural network (min-max [0,1])
            temp_normalized = self.preprocessor.normalize_temperature(temp)

            temperatures.append(temp_normalized)
            metadata.append({
                'original_shape': temp.shape,
                'orbit_type': swath['metadata'].get('orbit_type', 'unknown')
            })

            if (i + 1) % 100 == 0:
                gc.collect()

        data.close()
        self.logger.info(f"Loaded {len(temperatures)} temperature arrays")

        return temperatures, metadata


    def create_lr_hr_pairs(self, temperatures, scale_factor=2):
        """Create LR-HR pairs using SAME method as neural network"""
        lr_images = []
        hr_images = []
        valid_metadata = []

        self.logger.info("Creating LR-HR pairs using neural network method...")

        for i, temp_hr in enumerate(tqdm(temperatures, desc="Creating pairs")):
            # **CRITICAL CHANGE**: Use SAME LR creation method as neural network
            try:
                lr, hr = self.preprocessor.create_lr_hr_pair(temp_hr, scale_factor=scale_factor)

                lr_images.append(lr)
                hr_images.append(hr)
                valid_metadata.append(self.metadata[i])

            except Exception as e:
                self.logger.warning(f"Failed to create LR-HR pair for sample {i}: {e}")
                continue

        self.logger.info(f"Created {len(lr_images)} LR-HR pairs using neural network method")
        return lr_images, hr_images, valid_metadata

    def bicubic_upsample(self, lr_image, scale_factor):
        """Apply bicubic upsampling"""
        h, w = lr_image.shape
        target_h, target_w = h * scale_factor, w * scale_factor

        # Convert to tensor for PyTorch bicubic
        lr_tensor = torch.from_numpy(lr_image).unsqueeze(0).unsqueeze(0).float()

        # Apply bicubic interpolation
        sr_tensor = F.interpolate(lr_tensor,
                                  size=(target_h, target_w),
                                  mode='bicubic',
                                  align_corners=False)

        # Convert back to numpy
        sr_image = sr_tensor.squeeze().numpy()

        # **SAME RANGE AS NEURAL NETWORK**: clamp to [-1, 1] approximately
        # **FIXED RANGE**: clamp to [0, 1] for min-max normalization
        sr_image = np.clip(sr_image, 0, 1)

        return sr_image

    def calculate_metrics(self, sr_image, hr_image):
        """Calculate PSNR and SSIM"""
        # **CRITICAL**: Convert from [-1,1] to [0,255] for metrics calculation
        # This ensures fair comparison with neural network

        # First convert from [-1,1] to [0,1]
        # **FIXED**: Convert from [0,1] to [0,255] for metrics calculation
        # Data is already in [0,1] range from min-max normalization

        # Convert to [0,255] range for metrics calculation
        sr_uint8 = (sr_image * 255).astype(np.uint8)
        hr_uint8 = (hr_image * 255).astype(np.uint8)

        # Calculate PSNR
        psnr = calculate_psnr(sr_uint8, hr_uint8, crop_border=0, test_y_channel=False)

        # Calculate SSIM
        ssim = calculate_ssim(sr_uint8, hr_uint8, crop_border=0, test_y_channel=False)

        return psnr, ssim

    def save_comparison_image(self, lr_image, sr_image, hr_image, save_path, metadata):
        """Save clean comparison images without any scales or annotations"""
        # Create individual image files
        base_path = save_path.replace('.png', '')

        # **CONVERT BACK TO TEMPERATURE FOR VISUALIZATION**
        # Reverse the neural network normalization: T = norm * 150 + 200
        # **CONVERT BACK TO TEMPERATURE FOR VISUALIZATION**
        # Reverse the min-max normalization: need original min/max values
        # For now, use approximate temperature range
        lr_temp = lr_image * (350 - 50) + 50  # Approximate K range
        sr_temp = sr_image * (350 - 50) + 50
        hr_temp = hr_image * (350 - 50) + 50

        # Find common scale for consistent visualization
        vmin = min(lr_temp.min(), sr_temp.min(), hr_temp.min())
        vmax = max(lr_temp.max(), sr_temp.max(), hr_temp.max())

        # Create clean images without any annotations
        def save_clean_image(data, filename, colormap='turbo'):
            """Save image as clean array with exact dimensions"""
            fig, ax = plt.subplots(figsize=(data.shape[1] / 100, data.shape[0] / 100), dpi=100)
            ax.imshow(data, cmap=colormap, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.savefig(filename, dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close()

        # Save individual images in Kelvin
        save_clean_image(lr_temp, f'{base_path}_lr.png')
        save_clean_image(sr_temp, f'{base_path}_bicubic.png')
        save_clean_image(hr_temp, f'{base_path}_hr.png')

        # Create error map in Kelvin
        error = np.abs(sr_temp - hr_temp)
        save_clean_image(error, f'{base_path}_error.png', colormap='hot')

        # Also create a comparison grid (optional)
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # LR image
        im1 = axes[0].imshow(lr_temp, cmap='turbo', vmin=vmin, vmax=vmax)
        axes[0].set_title(f'LR ({lr_temp.shape[0]}×{lr_temp.shape[1]})')
        axes[0].axis('off')

        # Bicubic upsampled
        im2 = axes[1].imshow(sr_temp, cmap='turbo', vmin=vmin, vmax=vmax)
        axes[1].set_title(f'Bicubic SR ({sr_temp.shape[0]}×{sr_temp.shape[1]})')
        axes[1].axis('off')

        # HR ground truth
        im3 = axes[2].imshow(hr_temp, cmap='turbo', vmin=vmin, vmax=vmax)
        axes[2].set_title(f'HR Ground Truth ({hr_temp.shape[0]}×{hr_temp.shape[1]})')
        axes[2].axis('off')

        # Error map
        im4 = axes[3].imshow(error, cmap='hot')
        axes[3].set_title(f'Absolute Error (MAE: {error.mean():.2f} K)')
        axes[3].axis('off')

        # Calculate and display metrics
        psnr, ssim = self.calculate_metrics(sr_image, hr_image)
        plt.suptitle(f'PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f} | Temperature range: [{vmin:.0f}, {vmax:.0f}] K',
                     fontsize=14)

        plt.tight_layout()
        plt.savefig(f'{base_path}_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

        self.logger.info(f'Saved images for shape {hr_temp.shape}: {base_path}_*.png')

    def run_evaluation(self):
        """Run the complete evaluation"""
        # Load data
        npz_file = os.path.expandvars(self.config['data']['npz_file'])
        temperatures, self.metadata = self.load_temperature_data(
            npz_file,
            max_samples=self.config['evaluation']['max_samples']
        )

        # Create LR-HR pairs
        lr_images, hr_images, valid_metadata = self.create_lr_hr_pairs(
            temperatures,
            scale_factor=self.config['evaluation']['scale_factor']
        )

        if len(lr_images) > 0:
            self.logger.info(f"LR shape: {lr_images[0].shape}")
            self.logger.info(f"HR shape: {hr_images[0].shape}")
            self.logger.info(f"LR range: [{lr_images[0].min():.3f}, {lr_images[0].max():.3f}]")
            self.logger.info(f"HR range: [{hr_images[0].min():.3f}, {hr_images[0].max():.3f}]")

        # Create results directory with subdirectories
        os.makedirs('results/images', exist_ok=True)
        os.makedirs('results/images/clean_images', exist_ok=True)
        os.makedirs('results/images/comparisons', exist_ok=True)

        # Evaluate each pair
        self.logger.info("Starting bicubic evaluation with neural network preprocessing...")

        for i, (lr_img, hr_img, meta) in enumerate(tqdm(
                zip(lr_images, hr_images, valid_metadata),
                total=len(lr_images),
                desc="Evaluating"
        )):
            # Apply bicubic upsampling
            sr_img = self.bicubic_upsample(lr_img, self.config['evaluation']['scale_factor'])

            # Calculate metrics
            psnr, ssim = self.calculate_metrics(sr_img, hr_img)

            self.results['psnr_values'].append(psnr)
            self.results['ssim_values'].append(ssim)

            # Save first 10 images for comparison
            if i < 10:
                save_path = f'results/images/clean_images/image_{i:03d}.png'
                self.save_comparison_image(lr_img, sr_img, hr_img, save_path, meta)

            if (i + 1) % 100 == 0:
                self.logger.info(f"Processed {i + 1}/{len(lr_images)} images")
                gc.collect()

        # Calculate final statistics
        self.calculate_final_stats()

        # Save results
        self.save_results()

    def calculate_final_stats(self):
        """Calculate final statistics"""
        psnr_values = np.array(self.results['psnr_values'])
        ssim_values = np.array(self.results['ssim_values'])

        self.results['avg_psnr'] = np.mean(psnr_values)
        self.results['avg_ssim'] = np.mean(ssim_values)
        self.results['std_psnr'] = np.std(psnr_values)
        self.results['std_ssim'] = np.std(ssim_values)
        self.results['median_psnr'] = np.median(psnr_values)
        self.results['median_ssim'] = np.median(ssim_values)
        self.results['min_psnr'] = np.min(psnr_values)
        self.results['max_psnr'] = np.max(psnr_values)
        self.results['min_ssim'] = np.min(ssim_values)
        self.results['max_ssim'] = np.max(ssim_values)

        self.logger.info("=== BICUBIC INTERPOLATION RESULTS (NEURAL NETWORK PREPROCESSING) ===")
        self.logger.info(f"Number of images evaluated: {len(psnr_values)}")
        self.logger.info(f"Average PSNR: {self.results['avg_psnr']:.2f} ± {self.results['std_psnr']:.2f} dB")
        self.logger.info(f"Average SSIM: {self.results['avg_ssim']:.4f} ± {self.results['std_ssim']:.4f}")
        self.logger.info(f"Median PSNR: {self.results['median_psnr']:.2f} dB")
        self.logger.info(f"Median SSIM: {self.results['median_ssim']:.4f}")
        self.logger.info(f"PSNR range: [{self.results['min_psnr']:.2f}, {self.results['max_psnr']:.2f}] dB")
        self.logger.info(f"SSIM range: [{self.results['min_ssim']:.4f}, {self.results['max_ssim']:.4f}]")

    def save_results(self):
        """Save results to files"""
        # Save detailed results
        results_file = f'results/bicubic_neural_preprocessing_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.yaml'
        with open(results_file, 'w') as f:
            yaml.dump(self.results, f)

        # Create histogram plots
        self.plot_metric_distributions()

        self.logger.info(f"Results saved to {results_file}")

    def plot_metric_distributions(self):
        """Plot PSNR and SSIM distributions"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # PSNR histogram
        axes[0].hist(self.results['psnr_values'], bins=50, alpha=0.7, edgecolor='black')
        axes[0].axvline(self.results['avg_psnr'], color='red', linestyle='--',
                        label=f'Mean: {self.results["avg_psnr"]:.2f}')
        axes[0].axvline(self.results['median_psnr'], color='green', linestyle='--',
                        label=f'Median: {self.results["median_psnr"]:.2f}')
        axes[0].set_xlabel('PSNR (dB)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('PSNR Distribution (Neural Network Preprocessing)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # SSIM histogram
        axes[1].hist(self.results['ssim_values'], bins=50, alpha=0.7, edgecolor='black')
        axes[1].axvline(self.results['avg_ssim'], color='red', linestyle='--',
                        label=f'Mean: {self.results["avg_ssim"]:.4f}')
        axes[1].axvline(self.results['median_ssim'], color='green', linestyle='--',
                        label=f'Median: {self.results["median_ssim"]:.4f}')
        axes[1].set_xlabel('SSIM')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('SSIM Distribution (Neural Network Preprocessing)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/metric_distributions_neural_preprocessing.png', dpi=150, bbox_inches='tight')
        plt.close()


def main():
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Create evaluator and run
    evaluator = BicubicEvaluator(config)
    evaluator.run_evaluation()


if __name__ == '__main__':
    main()