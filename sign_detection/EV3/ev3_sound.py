"""
LEGO Mindstorms EV3 - sound
"""

# Copyright (C) 2016 Christoph Gaukel <christoph.gaukel@gmx.de>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import sign_detection.EV3.EV3 as ev3

my_ev3 = ev3.EV3(protocol=ev3.USB, host='00:16:53:47:92:46')
ops = b''.join([
    ev3.opSound,
    ev3.TONE,
    ev3.LCX(1),    # VOLUME
    ev3.LCX(440),  # FREQUENCY
    ev3.LCX(1000), # DURATION
])
my_ev3.send_direct_cmd(ops)
print ("Stuff")