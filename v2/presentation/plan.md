# Trade study
- scatterplot of all results, what you had earlier was pretty good
- make some actual system, like an input file, to designate scores for each parameter, that is then used to actually score all trajectories across both families, and generate the decision matrix of the top few choices. you can set some default weight scores for each category as you see fit for now, I can tweak those later. also add some params like min and max in this same file. like add a min departure dat of 2032, or a max c3 of 16 (so all of them pass), obviously you wont need a min c3, or a min delta v, but put the constraints that make sense in a file format that makes sense for this.

# Key facts about the background work
- make one markdown file with short slide-ready bullets on four things:
  - what optimization method we used
  - what the cloud workflow did for us
  - what scale of search we actually ran
  - what data we saved for each run / candidate
- make one compact quantitative summary output for the search campaign:
  - how many runs
  - how many candidate trajectories
  - total compute time if available
  - how many seeds / families were compared
- make one simple visual or table that explains the workflow at a glance:
  - config -> search -> archive -> trade study -> chosen trajectory analysis
- keep this section direct and audience-facing:
  - enough to explain the work clearly in a few slides
- NOTE: this should be number-driven, with the main goal to do a deep analysis of all the work that was done and the impressive quantities of time/compute/results that were produced.

# Chosen trajectory analysis
- Once we choose a trajectory from the trade study, we should probably do the following...
- Make a basic trajectory visualization GIF and 3D plot (some old script in scratch/visualize_candidate.py, idk if it's good/right tho)
- Make a *really* good visualization, maybe like following the spacecraft, showing it do the gravity assists , with a black background and like the planet textures, similar to GMAT visualizations. Stick to something, like a method/package that exists out there than making something from the ground up that'll take a long time to render this...
- Make a simple table of each DSM's values and info
- Make a more complex table showing orbital elements around each DSM and point of interest (launch and arrival included) that can then be taken and math can be done on those numbers.
