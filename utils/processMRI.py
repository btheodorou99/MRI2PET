# ADNI MRI SEARCH:
# Modality = MRI
# Weighting = "T1"
# Acquisition Plane = "Axial"
# Slice Thickness?

# DIRECTORY STRUCTURE:
# ADNI/
#   subjectID/
#     seriesName/
#       date_otherStuff/
#         someIDs/
#           dicomFiles

# MRI PROCESSING:
# dcm2niix -o tempdir/ ./dicomdir/
# import ants
# niix_file = "tempdir/" + os.listdir("tempdir/")[0][:-4] + ".nii"
# img = ants.image_read( "niix_file" )
# Registration? Maybe.
# Segmentation? Probably Not.
# img = ants.n4_bias_field_correction(img)
# data = img.numpy()
# data = ( data - data.min() )/( data.max()-data.min() )
# Save to NPY file and add metadata map to dataset
# img = ants.from_numpy( data )