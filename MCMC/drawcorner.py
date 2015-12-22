import corner

def make_figure(samples, true_vals):
    labels = [r"$R_{disk}$", r"$F_{disk}$", r"$\gamma_1^{disk}$", r"$\gamma_2^{disk}$",
              r"$R_{bulge}$", r"$F_{bulge}$", r"$\gamma_1^{bulge}$", r"$\gamma_2^{bulge}$",
              r"$\gamma_1^{shear}$", r"$\gamma_2^{shear}$"]
    figure = corner.corner(samples, labels=labels,
                           truths=true_vals,
                           show_titles=True)
    
    return figure
