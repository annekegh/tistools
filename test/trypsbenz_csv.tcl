# ============================
# COM of an unbinding molecule, loading PPTIS path folders
# Save as: com_unbinding_vmd.tcl
# Run in VMD:  vmd -e com_unbinding_vmd.tcl
# ============================

# ----------------------------
# USER SETTINGS
# ----------------------------
# 1) Reference topology / structure (use one that matches the xtc atom order)
#set TOPO "/run/user/1001/gvfs/smb-share:server=files.ugent.be,share=eliawils,user=eliawils/shares/tw06_biommeda_abl/paper-msm-short-term/simulations-and-analysis/trypsin-benzamidine/infrepptis/ext_input/conf.gro"                     ;# e.g., system.gro or system.pdb or system.psf (with coords)
set TOPO "/home/sina/Desktop/trypsin_infinit/trypsin_beta_2/gromacs_input/conf.gro"
set TOPO_TYPE "gro"                       ;# VMD will auto-detect in most cases; you can leave this empty ""

# 2) Base directory containing path folders and list of path numbers to load
set BASE_DIR "/home/sina/Desktop/trypsin_infinit/trypsin_beta_2/infrepptis4/load"
set SUBDIR "accepted"                     ;# Subdirectory within each path folder containing trajectory files

# 3) Classification CSV file and filter
set CSV_FILE "/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/APPTIS/tistools/test/classification_results.csv"
set FILTER_LABEL "A"                      ;# Set to "A" or "B" to filter paths by classification

# 4) Selections
set LIG_SEL  "resname BEN"                ;# selection for the unbinding molecule
set ALN_SEL  "protein and name CA"        ;# alignment selection, e.g., CA, or "backbone and noh"

# 5) Options
set DO_PBC_UNWRAP  0                      ;# 1 = fix PBC jumps (recommended); requires pbctools
set DO_PBC_WRAP    0                      ;# 1 = re-center system around protein after unwrapping
set DRAW_PATH      1                      ;# 1 = draw a polyline path of COM across frames (one per path number)
set WRITE_CSV      0                      ;# 1 = write COM coordinates to CSV (one per path number)
set WRITE_PDB      1                      ;# 1 = create a single-atom multi-model PDB for the COM trajectory
set OUT_CSV_PREFIX "COM_path"             ;# Prefix for CSV files (will append _PATHNR.csv)
set OUT_PDB        "COM_traj.pdb"

# ----------------------------
# HELPER: Read CSV and extract path numbers with specified label
# ----------------------------
proc read_classified_paths {csv_file filter_label} {
    set path_numbers {}
    
    if {[catch {set fh [open $csv_file "r"]} err]} {
        puts "ERROR: Cannot open CSV file: $csv_file ($err)"
        return $path_numbers
    }
    
    # Skip header line
    gets $fh header
    
    while {[gets $fh line] >= 0} {
        # Parse CSV line - handle quoted fields
        # Format: file,n_frames,method,label,confidence,dA,dB
        
        # Extract the label (4th field) - find it after the quoted file path
        if {[regexp {^"[^"]*",([^,]*),([^,]*),([^,]*),} $line -> n_frames method label]} {
            set label [string trim $label]
            
            if {$label eq $filter_label} {
                # Extract path number from the file path
                # Path format: .../load/PATHNR/PATHNR.xtc
                if {[regexp {/load/(\d+)/\d+\.xtc} $line -> pathnr]} {
                    lappend path_numbers $pathnr
                }
            }
        }
    }
    
    close $fh
    return $path_numbers
}

# ----------------------------
# Read path numbers from CSV based on classification
# ----------------------------
puts "Reading classifications from: $CSV_FILE"
puts "Filtering for label: $FILTER_LABEL"
set PATH_NUMBERS [read_classified_paths $CSV_FILE $FILTER_LABEL]
puts "Found [llength $PATH_NUMBERS] paths with label $FILTER_LABEL"

if {[llength $PATH_NUMBERS] == 0} {
    puts "ERROR: No paths found with label $FILTER_LABEL. Exiting."
    exit 1
}

# ----------------------------
# HELPER: Generate random color (RGB values 0-1)
# ----------------------------
proc random_color {} {
    set r [expr {rand()}]
    set g [expr {rand()}]
    set b [expr {rand()}]
    #return [list $r $g $b]
    return [list 0 0.5 1]
}

# ----------------------------
# HELPER: Find all XTC/TRR files in a directory
# ----------------------------
proc find_traj_files {dir} {
    set files {}
    if {[catch {glob -nocomplain -directory $dir *.xtc} xtc_files]} {
        set xtc_files {}
    }
    #if {[catch {glob -nocomplain -directory $dir *.trr} trr_files]} {
    #    set trr_files {}
    #}
    set files [concat $xtc_files]
    return [lsort $files]
}

# ----------------------------
# LOAD DATA
# ----------------------------
if {$TOPO_TYPE eq ""} {
    mol new $TOPO waitfor all
} else {
    mol new $TOPO type $TOPO_TYPE waitfor all
}

set topmol [molinfo top]

# Data structure to track paths: list of {path_number color frame_start frame_end com_list}
set path_data {}
set total_frames 0

foreach pathnr $PATH_NUMBERS {
    set path_dir [file join $BASE_DIR $pathnr $SUBDIR]
    puts "Processing path $pathnr from: $path_dir"
    
    # Find all trajectory files in this path directory
    set traj_files [find_traj_files $path_dir]
    
    if {[llength $traj_files] == 0} {
        puts "  WARNING: No XTC/TRR files found in $path_dir, skipping."
        continue
    }
    
    set frame_start $total_frames
    set path_color [random_color]
    
    # Load all trajectory files for this path
    foreach traj_file $traj_files {
        puts "  Loading: [file tail $traj_file]"
        mol addfile $traj_file waitfor all molid $topmol
    }
    
    set frame_end [expr {[molinfo $topmol get numframes] - 1}]
    set total_frames [expr {$frame_end + 1}]
    
    puts "  Path $pathnr: frames $frame_start to $frame_end ([expr {$frame_end - $frame_start + 1}] frames)"
    
    # Store path metadata
    lappend path_data [list $pathnr $path_color $frame_start $frame_end]
}

set nframes [molinfo $topmol get numframes]
puts "\nTotal frames loaded: $nframes"
puts "Total paths: [llength $path_data]\n"

# ----------------------------
# OPTIONAL: PBC FIXES
# ----------------------------
if {$DO_PBC_UNWRAP || $DO_PBC_WRAP} {
    if {[catch {package require pbctools} err]} {
        puts "WARNING: pbctools not available. Skipping PBC handling. ($err)"
    } else {
        # Unwrap to remove jumps (keeps molecules contiguous)
        if {$DO_PBC_UNWRAP} {
            # Unwrap per residue to avoid breaking residues at box boundaries
            pbc unwrap -all -sel "all" -compound residue
        }
        # Wrap back, centered on protein COM, so ligand stays near protein box
        if {$DO_PBC_WRAP} {
            set centersel $ALN_SEL
            pbc wrap -all -sel "all" -compound residue -center com -centersel "$centersel"
        }
    }
}

# ----------------------------
# ALIGNMENT (remove global rotation/translation)
# ----------------------------
set refsel [atomselect $topmol "$ALN_SEL" frame 0]
set movsel [atomselect $topmol "$ALN_SEL"]
set allsel [atomselect $topmol "all"]

for {set i 0} {$i < $nframes} {incr i} {
    $movsel frame $i
    # Best-fit transform mapping current alignment atoms to reference (frame 0)
    set M [measure fit $movsel $refsel]
    $allsel frame $i
    $allsel move $M
}
$refsel delete
$movsel delete
$allsel delete

# ----------------------------
# COMPUTE LIGAND COM PER FRAME FOR EACH PATH
# ----------------------------
set ligsel [atomselect $topmol "$LIG_SEL"]
set timestep_ps 0.02  ;# time interval in picoseconds

# Process each path separately
set path_data_with_coms {}
foreach path_info $path_data {
    set pathnr [lindex $path_info 0]
    set path_color [lindex $path_info 1]
    set frame_start [lindex $path_info 2]
    set frame_end [lindex $path_info 3]
    
    puts "Computing COM for path $pathnr (frames $frame_start-$frame_end)..."
    
    set com_list {}
    set lig_missing 0
    
    for {set i $frame_start} {$i <= $frame_end} {incr i} {
        $ligsel frame $i
        if {[llength [$ligsel get index]] == 0} {
            lappend com_list [list $i "" "" "" ""]
            incr lig_missing
            continue
        }
        
        set com [measure center $ligsel weight mass]
        set x [lindex $com 0]
        set y [lindex $com 1]
        set z [lindex $com 2]
        
        if {$timestep_ps != 0.0} {
            set t [expr {$i * $timestep_ps}]
        } else {
            set t ""
        }
        lappend com_list [list $i $t $x $y $z]
    }
    
    if {$lig_missing > 0} {
        puts "  WARNING: Ligand empty in $lig_missing frames for path $pathnr"
    }
    
    # Store path info with COM data
    lappend path_data_with_coms [list $pathnr $path_color $com_list]
}

$ligsel delete

# ----------------------------
# WRITE CSV (one file per path)
# ----------------------------
if {$WRITE_CSV} {
    foreach path_info $path_data_with_coms {
        set pathnr [lindex $path_info 0]
        set com_list [lindex $path_info 2]
        
        set out_file "${OUT_CSV_PREFIX}_${pathnr}.csv"
        set fh [open $out_file "w"]
        puts $fh "frame,time_ps,x_nm,y_nm,z_nm"
        
        foreach rec $com_list {
            set frame [lindex $rec 0]
            set time  [lindex $rec 1]
            set x     [lindex $rec 2]
            set y     [lindex $rec 3]
            set z     [lindex $rec 4]
            puts $fh "$frame,$time,$x,$y,$z"
        }
        close $fh
        puts "Wrote COM CSV for path $pathnr -> $out_file"
    }
}

# ----------------------------
# DRAW 3D PATHS (one polyline per path with unique color)
# ----------------------------
if {$DRAW_PATH} {
    graphics $topmol delete all
    
    set color_id 0
    foreach path_info $path_data_with_coms {
        set pathnr [lindex $path_info 0]
        set path_color [lindex $path_info 1]
        set com_list [lindex $path_info 2]
        
        # Assign a unique color ID and set its RGB
        color change rgb $color_id [lindex $path_color 0] [lindex $path_color 1] [lindex $path_color 2]
        graphics $topmol color $color_id
        
        puts "Drawing path $pathnr with color [lindex $path_color 0] [lindex $path_color 1] [lindex $path_color 2]"
        
        # Draw each frame as a sphere (dot)
        foreach rec $com_list {
            set x [lindex $rec 2]
            set y [lindex $rec 3]
            set z [lindex $rec 4]
            if {$x eq ""} { continue }
            
            set pos [list $x $y $z]
            graphics $topmol sphere $pos radius 0.15 resolution 10
        }
        
        incr color_id
        if {$color_id >= 1024} { set color_id 0 }  ;# wrap around if too many paths
    }
    puts "\nDrew [llength $path_data_with_coms] COM paths as dots in VMD display."
}

# ----------------------------
# OPTIONAL: WRITE A SINGLE-ATOM MULTI-MODEL PDB (all paths combined)
# ----------------------------
if {$WRITE_PDB} {
    set fh [open $OUT_PDB "w"]
    set model 1
    
    foreach path_info $path_data_with_coms {
        set pathnr [lindex $path_info 0]
        set com_list [lindex $path_info 2]
        
        foreach rec $com_list {
            set x [lindex $rec 2]
            set y [lindex $rec 3]
            set z [lindex $rec 4]
            if {$x eq ""} {
                incr model
                continue
            }
            puts $fh [format "MODEL     %4d" $model]
            puts $fh [format "HETATM%5d  COM COM A%4d    %8.3f%8.3f%8.3f  1.00  0.00           C" 1 $pathnr $x $y $z]
            puts $fh "ENDMDL"
            incr model
        }
    }
    close $fh
    puts "\nWrote COM single-atom trajectory -> $OUT_PDB"
    puts "Tip: Load it in VMD (mol new $OUT_PDB), then color/represent as a sphere, and play."
}

puts "Done."
