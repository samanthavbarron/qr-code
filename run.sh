rm *.png || :
rm images/*.png || :
conda activate qrcode
python qr_code.py
