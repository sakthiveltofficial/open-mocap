# Contributing

Thanks for your interest in Open Mocap. Issues and pull requests are welcome.

## Reporting bugs

Open an issue with:

- your OS, Blender version, and GPU (if any),
- the engine you used (Fast or Pro),
- what you did and what happened,
- the console output (Blender's system console shows the capture output) and, for install problems, `installer/install.log`.

A short clip that reproduces a tracking problem helps a lot, if you can share one.

## Pull requests

1. Fork the repository and create a branch from `main`.
2. Keep changes focused; one fix or feature per pull request.
3. Test what you touched:
   - capture code: run `save_mp_data.py` (or `save_mhr_data.py`) on a short clip and `npz_to_bvh.py` on the result,
   - add-on code: enable the add-on in Blender 4.2 or newer and exercise the panel or operator.
4. Describe what changed and how you tested it.

Both engines write the same NPZ layout and skeleton (see `pipeline/mp_skeleton.py` and `pipeline/body_skeleton.py`). Changes to one engine's output should keep that contract so presets and cleanup keep working for both.

## License

Open Mocap is GPL-3.0-or-later. By contributing you agree that your contribution is licensed under the same terms. Don't add model weights or other large binaries to the repository; the installer downloads models from their official sources.

## Security

Please don't report security issues in public. Use GitHub's private vulnerability reporting (the **Security** tab, **Report a vulnerability**).
