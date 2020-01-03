
# Install

The best way to install `LogMl` is to create a virtual environment and pull the code from GitHub:
- Development version: `master` branch
- Latest Release version: `release-YYYYMMDDHHmmSS` (release versions are identified by a timestamp: year, month, day, hour, minute, second)

### Requirements

- Operating system: `LogMl` was developed and tested on Linux and Mac (OS.X).
- Python 3.7
- Virtual environment
- `pip`

### How to install

Open a shell / terminal and follow these steps:

```
# This creates a 'source' directory and pulls the latest (development) version from GitHub
SRC_DIR="$HOME/workspace"
mkdir -p $SRC_DIR
cd $SRC_DIR
git clone https://github.com/AstraZeneca-NGS/LogMl.git

# Running install.sh will install LogMl into the default directory ($HOME/logml)
cd LogMl
./scripts/install.sh
```

The `scripts/install.sh` script should take care of installing in a default directory (`$HOME/logml`).
If you want another directory, just edit the script and change the `INSTALL_DIR` variable in the script.
