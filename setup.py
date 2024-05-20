from setuptools import setup

exec(open('./version.py').read())

setup(
  name = 'var_ms_imggen',
  package_dir = {
    'data': 'data',
    'model': 'model',
  },
  version = __version__,
  description = 'image gen from VAR - Pytorch',
  long_description_content_type = 'text/markdown',
  author = 'liruoyu01',
  author_email = 'liruoyu@in.ai',
  url = 'https://github.com/liruoyu01/var_ms_imggen',
  keywords = [
    'genai',
    'transformer',
    'generative image model'
  ],
  install_requires=[
    'accelerate>=0.24.0',
    'einops>=0.7.0',
    'ema-pytorch>=0.2.4',
    'pytorch-warmup',
    'opencv-python',
    'pillow',
    'numpy',
    'torch',
    'torchvision',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Programming Language :: Python :: 3.6',
  ],
)
