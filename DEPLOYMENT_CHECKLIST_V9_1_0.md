# Deployment Checklist v9.1.0

1. Hapus file produksi versi lama dari root repository.
2. Salin isi paket deploy-only langsung ke root repository.
3. Pastikan `app.py` menjadi Main file path.
4. Pastikan `database_first.py` ikut ter-deploy.
5. Pertahankan Streamlit Secrets; jangan commit secret.
6. Jalankan migration database yang sudah tersedia bila tabel Supabase belum ada.
7. Jalankan mode `Isi Database` dengan 80 ticker per proses.
8. Ulangi sampai status `READY_FOR_DAILY_SCAN`.
9. Baru gunakan `Scan Harian`.
10. Periksa audit provider dan database setelah proses pertama.
