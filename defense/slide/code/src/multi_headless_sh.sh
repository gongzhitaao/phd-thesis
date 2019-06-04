#!/bin/bash

export GPUS=1
export FIRST_XORG_SERVER=200
export FIRST_VIRTUAL_DISPLAY=8

# for ((gpu=0; gpu<${GPUS}; gpu++)) do
#     # Start X.Org server.
#     export xorg_server=$((FIRST_XORG_SERVER + gpu))
#     echo ${xorg_server}
#     nohup Xorg :${xorg_server} -config xorg.conf.${gpu} &

#     # Start VNC server.
#     export virtual_display=$((xorg_server + GPUS))
#     echo ${virtual_display}
#     nohup /opt/TurboVNC/bin/vncserver :${virtual_display}

#     # Set DISPLAY.
#     export DISPLAY=:${virtual_display}

#     # Should print the OpenGL version.
#     # vglrun -d :${xorg_server}.0 glxinfo | grep "OpenGL version"
# done

for ((gpu=0; gpu<${GPUS}; gpu++)) do
    export xorg_server=$((FIRST_XORG_SERVER + gpu))
    export virtual_display=$((xorg_server + GPUS))
    export DISPLAY=:${virtual_display}

    # nohup vglrun -d :${xorg_server}.0 python generate_poses.py ${gpu} > ${gpu}.log &
    nohup vglrun -d :${xorg_server}.0 python renderer_test.py > ${gpu}.log &
done
