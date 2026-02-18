def joint_name_seqikpy2flygym(seqikpy_name):
    side = seqikpy_name[0]
    legpos = seqikpy_name[1]
    if side not in "LR" or legpos not in "FMH":
        raise ValueError(f"Unexpected SeqIKPy joint name {seqikpy_name}")

    flygym_leg = f"{side}{legpos}".lower()
    seqikpy_anatomical_joint = seqikpy_name.split("_")[0][2:]
    if seqikpy_anatomical_joint == "ThC":
        parent = "c_thorax"
        child = f"{flygym_leg}_coxa"
    elif seqikpy_anatomical_joint == "CTr":
        parent = f"{flygym_leg}_coxa"
        child = f"{flygym_leg}_trochanterfemur"
    elif seqikpy_anatomical_joint == "FTi":
        parent = f"{flygym_leg}_trochanterfemur"
        child = f"{flygym_leg}_tibia"
    elif seqikpy_anatomical_joint == "TiTa":
        parent = f"{flygym_leg}_tibia"
        child = f"{flygym_leg}_tarsus1"
    else:
        raise ValueError(f"Unknown SeqIKPy anatomical joint {seqikpy_anatomical_joint}")

    axis = seqikpy_name.split("_")[1]
    if axis not in ("pitch", "roll", "yaw"):
        raise ValueError(f"Unexpected axis {axis} in SeqIKPy joint name {seqikpy_name}")

    flygym_joint_name = f"{parent}-{child}-{axis}"
    return flygym_joint_name
