
import numpy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

def plot_Crit( Crit2, name, **kwargs ) :
    svalues = name.encode()
    sids = (name + "_trackIds").encode()
    fake = Crit2[ svalues ][ Crit2[ sids ] == -1 ]
    real = Crit2[ svalues ][ (Crit2[ sids ] > -1)]

    fs = ( 8, 4.5 )
    fs = kwargs.get('fs', fs)
    
    rmin = kwargs.get( "min", numpy.amin( Crit2[ svalues ].flatten() ) )
    rmax = kwargs.get( "max", numpy.amax( Crit2[ svalues ].flatten() ) )
    nbins = kwargs.get( "nbins", (rmax-rmin) * 5 )

    plt.figure(figsize=fs, dpi=200)
    plt.hist( fake.flatten(), bins=numpy.linspace(rmin, rmax, nbins ), alpha=0.75, histtype='stepfilled' )
    plt.hist( real.flatten(), bins=numpy.linspace(rmin, rmax, nbins ), alpha=0.75, histtype='stepfilled' )
    plt.legend( ["Fake", "Real"], loc='upper right', prop={'size': 10})
    plt.semilogy()
    plt.ylabel('N')
    plt.xlabel(name)
    plt.savefig( "fig_" + name + ".png", dpi = 100 )
    plt.show()
    

def plot_Crit_corr( Crit2, name1, name2, **kwargs ) :

    sv1 = name1.encode()
    si1 = (name1 + "_trackIds").encode()
    fake1 = Crit2[ sv1 ][ Crit2[ si1 ] == -1 ]
    real1 = Crit2[ sv1 ][ (Crit2[ si1 ] > -1)]

    sv2 = name2.encode()
    si2 = (name2 + "_trackIds").encode()
    fake2 = Crit2[ sv2 ][ Crit2[ si2 ] == -1 ]
    real2 = Crit2[ sv2 ][ (Crit2[ si2 ] > -1)]

    show_fake = kwargs.get( 'fake', False )

    
    x = real1.flatten()
    y = real2.flatten()
    c = 'r'

    if show_fake :
        c = 'b'
        x = fake1.flatten()
        y = fake2.flatten()
        print( "FAKE" )
    
    plt.scatter( x, y, s=0.15, c=c, alpha=1 )
    
    plt.ylabel(name2)
    plt.xlabel(name1)

def eff_purity( real, fake, cmin, cmax ) :
    rni = numpy.where(numpy.logical_and(real>=cmin, real<=cmax))[0]
    fin = numpy.where(numpy.logical_and(fake>=cmin, fake<=cmax))[0]

    # print( rni )

    eff = rni.size / real.size
    if rni.size + fin.size > 0 :
        purity = (rni.size / ( rni.size + fin.size ))
    else:
        purity = 0
    return (eff, purity)

def find_optimal_max( Crit2, name, stepsize, **kwargs ) :
    svalues = name.encode()
    sids = (name + "_trackIds").encode()
    fake = Crit2[ svalues ][ Crit2[ sids ] == -1 ].flatten()
    real = Crit2[ svalues ][ (Crit2[ sids ] > -1)].flatten()

    fs = ( 8, 4.5 )
    fs = kwargs.get('fs', fs)

    rmin = kwargs.get( "min", numpy.amin( Crit2[ svalues ].flatten() ) )
    rmax = kwargs.get( "max", numpy.amax( Crit2[ svalues ].flatten() ) )

    cmin = kwargs.get( "cmin", rmin )

    effs = []
    puritys = []
    highcuts = []

    target_eff = kwargs.get( "target_eff", 0.95 )
    target_purity = kwargs.get( "target_purity", 0.9 )
    last_eff = 0
    last_purity = 0
    for hc in numpy.arange( cmin, rmax, stepsize ) :
        (eff, purity) = eff_purity( real, fake, cmin, hc )
        effs.append( eff )
        puritys.append( purity )
        highcuts.append( hc )

        if last_eff < target_eff and eff >= target_eff :
            print( "Target Efficiency of %0.2f @ cut max = %f (purity = %0.2f)" %(target_eff, hc, purity) )

        if last_purity < target_purity and purity >= target_purity :
            print( "Target Purity of %0.2f @ cut max = %f (Efficiency = %0.2f)" % (target_purity, hc, eff) )

        last_eff = eff
        last_purity = purity

    fig = plt.figure(figsize=fs, dpi=200)
    sc0 = plt.scatter( highcuts, puritys, s=0.5 )
    sc1 = plt.scatter( highcuts, effs, s=0.5 )
    plt.xlabel( 'cut max'  )
    plt.ylabel( ''  )
    plt.margins(x=0, y=0.05)
    lgnd = plt.legend( ["Purity", "Efficiency"], loc='center right', prop={'size': 10})
    lgnd.legendHandles[0]._sizes = [60]
    lgnd.legendHandles[1]._sizes = [60]
    # plt.plot([rmin, rmax], [1.0, 1.0], 'k:', lw=2)
    
    plt.plot([rmin, rmax], [target_eff, target_eff], "k:", lw=1)
    plt.annotate( "Target Efficiency ", xy=(rmin + 0.6 * ( rmax - rmin), target_eff - 0.05), color=sc1.get_facecolors()[0] )
    

    plt.show()



def find_optimal_window( Crit2, name, stepsize, **kwargs ) :
    svalues = name.encode()
    sids = (name + "_trackIds").encode()
    fake = Crit2[ svalues ][ Crit2[ sids ] == -1 ].flatten()
    real = Crit2[ svalues ][ (Crit2[ sids ] > -1)].flatten()

    fs = ( 8, 4.5 )
    fs = kwargs.get('fs', fs)
    

    rmin = kwargs.get( "min", numpy.amin( Crit2[ svalues ].flatten() ) )
    rmax = kwargs.get( "max", numpy.amax( Crit2[ svalues ].flatten() ) )

    plot3d = kwargs.get( "proj3d", False )
    

    effs = []
    puritys = []
    lowcuts = []
    highcuts = []
    cut_options = []

    target_eff = kwargs.get( "target_eff", 0.95 )
    
    for hc in numpy.arange( rmin - stepsize*2, rmax, stepsize ) :
        last_eff = 0
        for lc in numpy.arange( rmin - stepsize * 2, rmax, stepsize ) :
            
            (eff, purity) = eff_purity( real, fake, lc, hc )
            if last_eff < target_eff and eff >= target_eff :
                cut_options.append( (lc, hc, purity, eff) )
            # remove some bad results
            if eff < 0.02 : 
                purity = 0.0 
            effs.append( eff )
            puritys.append( purity )
            lowcuts.append( lc )
            highcuts.append( hc )
            # print( "%f to %f = (eff=%f, purity=%f)" %(lc, hc, eff, purity) )

    best_options = sorted( 
        sorted( cut_options, key=lambda x:x[0], reverse=True)
        , key=lambda x: x[2], reverse=True )
    
    for opt in best_options[0:10] :
        print( "Target Efficiency of %0.2f @ cut (%f, %f) (purity = %0.3f)" %(opt[3], opt[0], opt[1], opt[2]) )

    # plt.scatter( puritys, effs )
    # plt.xlim([0, 1.0])

    x = numpy.asarray(lowcuts)
    y = numpy.asarray(highcuts)
    z0 = numpy.asarray(puritys)
    z1 = numpy.asarray(effs)

    x=numpy.unique(x)
    y=numpy.unique(y)
    X,Y = numpy.meshgrid(x,y)

    Z=z0.reshape(len(y),len(x))

    if plot3d == True:
        fig = plt.figure(figsize=fs, dpi=200)
        axes = [0]*2
        axes[0] = fig.add_subplot( 1, 2, 1, projection='3d' )
        if 'angle0' in kwargs  :
            angle0 = kwargs['angle0']
            axes[0].view_init( angle0[0], angle0[1] )
        axes[1] = fig.add_subplot( 1, 2, 2, projection='3d' )
        if 'angle1' in kwargs  :
            angle1 = kwargs['angle1']
            axes[1].view_init( angle1[0], angle1[1] )
        axes[0].plot_surface(X,Y,Z, cmap=cm.viridis, antialiased=True)
        Z=z1.reshape(len(y),len(x))
        pcm = axes[1].plot_surface(X,Y,Z, cmap=cm.viridis, antialiased=True)
    else :
        nn = colors.LogNorm(vmin=1e-1, vmax=1.0)
        nn = colors.Normalize( 0, Z.max() )
        fig, axes = plt.subplots(1, 2, figsize=fs, dpi=200)
        fig.tight_layout(pad=3.0)
        pcm = axes[0].pcolormesh(X,Y,Z, norm=nn, cmap=cm.viridis)
        plt.colorbar( pcm, ax = axes[0])
        Z=z1.reshape(len(y),len(x))
        nn = colors.Normalize( 0, Z.max() )
        pcm = axes[1].pcolormesh(X,Y,Z, norm=nn, cmap=cm.viridis)
        plt.colorbar( pcm, ax = axes[1])

    # plt.colorbar( pcm, ax = axes[1])
    axes[0].set_xlabel( 'cut min'  )
    axes[0].set_ylabel( 'cut max'  )

    axes[1].set_xlabel( 'cut min'  )
    # axes[1].set_ylabel( 'cut max'  )

    plt.show()
    # rni = np.where(np.logical_and(real>=cmin, real<=cmax))
    # fin = np.where(np.logical_and(real>=cmin, real<=cmax))