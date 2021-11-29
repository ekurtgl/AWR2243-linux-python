import subprocess
from getpass import getpass

sudo_password = '190396'
build_dca = '/home/emre/Desktop/77ghz/open_radar/open_radar_initiative-new_receive_test/' \
            'open_radar_initiative-new_receive_test/setup_dca_1000/build'

build_radar = '/home/emre/Desktop/77ghz/open_radar/open_radar_initiative-new_receive_test/' \
              'open_radar_initiative-new_receive_test/setup_radar/build'

setup_dca = 'sudo ./setup_dca_1000'
setup_radar = 'sudo ./setup_radar'
trig_dca = setup_dca.split()
trig_radar = setup_radar.split()

subprocess.run(trig_dca, cwd=build_dca)
subprocess.run(trig_radar, cwd=build_radar)

