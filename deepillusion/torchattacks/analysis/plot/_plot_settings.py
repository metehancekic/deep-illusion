# these settings enable LaTeX in matplotlib and make the figures pretty


from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

colors = [(1, 0, 0), (1, 1, 1), (0, 0, 1)]
cm = LinearSegmentedColormap.from_list("my_list", colors)

rcParams["figure.dpi"] = 100
rcParams["figure.figsize"] = [6.0, 4.0]
rcParams["axes.linewidth"] = 0.5
rcParams["font.size"] = 11.0

axisFace = "#323A48"
figureFace = "#323A48"
textColor = "#DBE1EA"
edgeColor = "#92A2BD"
gridColor = "#3F495A"
notebook_bg = "#1A2028"
yellow = "#FFEC8E"
orange = "#ff7f0e"
red = "#e17e85"
magenta = "#e07a7a"
violet = "#be86e3"
blue = "#1f77b4"
cyan = "#4cb2ff"
green = "#61ba86"
rcParams["lines.linewidth"] = 1.25

# Latex related settings

rc("text", usetex=True)
rc(
    "text.latex",
    preamble=r"\usepackage{amsmath}   \usepackage{mathrsfs} \usepackage{amssymb}",
)
rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"
