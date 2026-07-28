# The following code will only execute
# successfully when compression is complete

import kagglehub

# Download latest version
kagglehub.login()
path = kagglehub.competition_download("filament-segmentation-2026")

print("Path to competition files:", path)
