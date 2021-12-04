import plotly as py
import plotly.graph_objs as go
import ipywidgets as widgets
import numpy as np

def shape_fig(brdf_str, model_str, case_str, it_no):
    
    xyz = np.load('results/{}-{:02}-coords.npy'.format(brdf_str+'-'+case_str+'-'+model_str, it_no))
    normal = np.load('results/{}-{:02}-normals.npy'.format(brdf_str+'-'+case_str+'-'+model_str, it_no))
    
    x,y,z = xyz[:,0],xyz[:,1],xyz[:,2]

    surface = go.Scatter3d(x=x.flatten(), 
                           y=y.flatten(), 
                           z=z.flatten(),
                           mode='markers',
                           marker = {"size": 1, "color": ((normal+1)/2)})
    data = [surface]

    layout = go.Layout(
        scene=dict(
            xaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            yaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            zaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            )
        )
    )

    fig = go.Figure(data=data, layout=layout)
    return fig

def shape_widget_all(brdf_str, model_str, case_str, it_no, sync_views=True):
    
    children = []
    colors = []
    i = 1
    i_s = []
    min_depth, max_depth = np.load('{}-depth_range.npy'.format(model_str))
    while i <= it_no:
        children.append(go.FigureWidget(shape_fig(brdf_str, model_str, case_str, i-1)))
        depth = np.load('results/{}-{:02}-depths.npy'.format(brdf_str+'-'+case_str+'-'+model_str, i-1))
        gt_depth = np.load('results/{}-{:02}-gt_depths.npy'.format(brdf_str+'-'+case_str+'-'+model_str, i-1))
        normal = np.load('results/{}-{:02}-normals.npy'.format(brdf_str+'-'+case_str+'-'+model_str, i-1))
        gt_normal = np.load('results/{}-{:02}-gt_normals.npy'.format(brdf_str+'-'+case_str+'-'+model_str, i-1))
        depth = (depth - min_depth) / (max_depth - min_depth)
        gt_depth = (gt_depth - min_depth) / (max_depth - min_depth)
        color = {
            'Est. Normal': (normal + 1) / 2,
            'Est. Depth': np.clip(depth, 0, 1),
            'Depth Err.': np.clip(np.abs(depth-gt_depth), 0, 1),
            'Normal Err.': np.clip(np.arccos(np.clip((normal * gt_normal).sum(-1), 0, 1))*180/np.pi, 0, 40),
        }
        colors.append(color)
        i_s.append(i)
        if i*2 > it_no:
            if i != it_no:
                i = it_no
                continue
        i = i * 2
    
    CAN_LOCK = [True]
    def cam_change(layout, camera):
        if CAN_LOCK[0]:
            CAN_LOCK[0] = False
            for f in children:
                f.layout.scene.camera = camera
            CAN_LOCK[0] = True
    if sync_views:
        for f in children:
            f.layout.scene.on_change(cam_change, 'camera')

    tab = widgets.Tab()
    tab.children = children
    for i, j in zip(range(len(children)), i_s):
        tab.set_title(i, 'iteration #{:02}'.format(j))
    
    button = widgets.ToggleButtons(
    options=['Est. Normal', 'Est. Depth', 'Normal Err.', 'Depth Err.', ],
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltips=['Points are colored estimated normal',
              'Points are colored by estimated depth, [min_depth, max_depth] normalised to [0,1]', 
              'Points are colored by normal angular errors',
              'Points are colored by absolute depth errors, [min_depth, max_depth] normalised to [0,1]',  ],
    #     icons=['check'] * 3
    )
    A = None
    def fun(arg):
        for i in range(len(children)):
            children[i].data[0]["marker"]['color'] = colors[i][arg['new']]
            if arg['new'] in ['Est. Depth', 'Normal Err.', 'Depth Err.']:
                children[i].data[0]["marker"]['showscale'] = True
            else:
                children[i].data[0]["marker"]['showscale'] = False

    button.observe(fun, 'value')
    
    return widgets.VBox([button,tab])

import matplotlib.pyplot as plt
import cv2


def rerender(brdf_str, model_str, case_str, it_no, sync_views=True):
    i_s = []
    i = 1
    while i <= it_no:
        i_s.append(i-1)
        if i*2 > it_no:
            if i != it_no:
                i = it_no
                continue
        i = i * 2
    n_imgs = [cv2.imread('results/{}-{:02}-n.png'.format(brdf_str+'-'+case_str+'-'+model_str, i), -1)[::-1, ::-1 ,::-1] for i in i_s]
    l_imgs = [cv2.imread('results/{}-{:02}-l.png'.format(brdf_str+'-'+case_str+'-'+model_str, i), -1)[::-1, ::-1 ,::-1] for i in i_s]
    
    
    
    #gtn = cv2.imread('results/{}-00-gtn.png'.format(brdf_str+'-'+case_str+'-'+model_str))[::-1, ::-1 ,::-1]
    fig, axes = plt.subplots(2, len(i_s), sharex=True, sharey=True,figsize=(13.5,4.5))
    for i in range(len(i_s)):
        axes[0,i].imshow(n_imgs[i])
        axes[1,i].imshow(l_imgs[i])
        axes[0,i].title.set_text('Iteration {:02}'.format(i_s[i]+1))
        axes[0,i].axis('off')
        axes[1,i].axis('off')
    #axes[2, len(i_s)//2-1].imshow()
    plt.subplots_adjust(left=0,right=1, wspace=0.02, hspace=0)
    fig.suptitle('Normal maps and rerendered img (with diffuse and co-located)')
    plt.axis('off')
    plt.show(fig)
    
def inputs(brdf_str, model_str, case_str, no_views=30, sync_views=False):
    i_imgs = [cv2.imread('results/{}-{:02}-r.png'.format(brdf_str+'-'+case_str+'-'+model_str, i), -1)[::-1, ::-1 ,::-1] for i in range(no_views)]
  
    #gtn = cv2.imread('results/{}-00-gtn.png'.format(brdf_str+'-'+case_str+'-'+model_str))[::-1, ::-1 ,::-1]
    fig, axes = plt.subplots(6, 5, sharex=False, sharey=False,figsize=(13,15.5))
    for i in range(no_views):
        axes[i//5,i%5].imshow(i_imgs[i])
        axes[i//5,i%5].axis('off')
    #axes[2, len(i_s)//2-1].imshow()
    plt.subplots_adjust(left=0,right=1, wspace=0.02, hspace=0.02)
    fig.suptitle('Input images')
    plt.show(fig)
    
def convergence(brdf_str, model_str, case_str):
    brdf = np.load('results/{}-brdf-mae.npy'.format(brdf_str+'-'+case_str+'-'+model_str))
    depth = np.load('results/{}-depth-rms.npy'.format(brdf_str+'-'+case_str+'-'+model_str))
    normal = np.load('results/{}-normal-mean.npy'.format(brdf_str+'-'+case_str+'-'+model_str))
    
    f1 = go.Figure(data=go.Scatter(x=np.arange(30), y=brdf, name='BRDF mean relat. error', mode='lines+markers'), layout=go.Layout(dict(title='BRDF mean relat. error', xaxis_title='iterations', yaxis_title='error', width=380)))
    f2 = go.Figure(data=go.Scatter(x=np.arange(30), y=depth, name='Depth RMS\n([0,max] normalised to [0,1])', mode='lines+markers'), layout=go.Layout(dict(title='Depth RMS ([0,max] normalised to [0,1])', xaxis_title='iterations', yaxis_title='error', width=420)))
    f3 = go.Figure(data=go.Scatter(x=np.arange(30), y=normal, name='normal mean angular error (degrees)', mode='lines+markers'), layout=go.Layout(dict(title='normal mean angular error (degrees)', xaxis_title='iterations', yaxis_title='error', width=400)))
    
    return widgets.HBox([go.FigureWidget(f1), go.FigureWidget(f2), go.FigureWidget(f3)])

from utils import brdf_interp_2d, brdf_interp, load_brdf_2d, load_brdf
import plotly.express as px
def BRDF(brdf_str, model_str, case_str, it_no):
    
    i_s = []
    i = 1
    while i <= it_no:
        i_s.append(i-1)
        if i*2 > it_no:
            if i != it_no:
                i = it_no
                continue
        i = i * 2
    figs = []
    for i in i_s:
        # scattered_data = np.load('results/{}-{:02}-scatter.npz'.format(brdf_str+'-'+case_str+'-'+model_str, i))
        if case_str == '2d':
            brdf_slice = np.load('results/{}-{:02}-brdf.npy'.format(brdf_str+'-'+case_str+'-'+model_str, i))
            brdf_slice = np.swapaxes(np.swapaxes(brdf_slice.squeeze(0), 0, 1), 1, 2) # [theta_h, theta_d, 3]
            gt_brdf_slice = load_brdf_2d(brdf_str, log=True).gpu.img.cpu().numpy()
            gt_brdf_slice = np.swapaxes(np.swapaxes(gt_brdf_slice.squeeze(0), 0, 1), 1, 2)
            gt_brdf_slice = np.swapaxes(gt_brdf_slice, 0,1)
            # visualising in 3d
#             figs.append(go.Figure(data=[
# #                                         go.Surface(z=brdf_slice[...,0], name='R'),
#                                         go.Surface(z=brdf_slice[...,1], name='G'),
# #                                         go.Surface(z=brdf_slice[...,2], name='B'),
#                                        ], layout=go.Layout(dict(title='iteration {:02}'.format(i+1)))))
            # visualising in 2d slice
            brdf_slice = np.swapaxes(brdf_slice, 0,1)
            fig = px.imshow(brdf_slice, zmin=-6, zmax=4, origin='lower', range_color=[-6,4], width=400, height=400, title='iteration {:02}'.format(i))
            fig.update_traces(hovertemplate="h: %{x}<br>d: %{y}<br>R:%{meta[0]:.2f};GT%{meta[3]:.2f}<br>G:%{meta[1]:.2f};GT%{meta[4]:.2f}<br>B:%{meta[2]:.2f};GT%{meta[5]:.2f}", meta=np.concatenate([brdf_slice,gt_brdf_slice],axis=-1))
            figs.append(fig)
        
        elif case_str == '1d':
            brdf_func = brdf_interp.load('results/{}-{:02}-brdf.npy'.format(brdf_str+'-'+case_str+'-'+model_str, i))
            x = np.arange(90)
            y = brdf_func(np.cos(x/180*np.pi))
            fig = go.Figure(data=[go.Scatter(x=x, y=y[...,0],name='R'),go.Scatter(x=x, y=y[...,1],name='G'),go.Scatter(x=x, y=y[...,2],name='B')], layout=go.Layout(dict(title='iteration {:02}'.format(i))) )
            figs.append(fig)
    
    if case_str == '2d':
        brdf_slice = load_brdf_2d(brdf_str, log=True).gpu.img.cpu().numpy()
        brdf_slice = np.swapaxes(np.swapaxes(brdf_slice.squeeze(0), 0, 1), 1, 2) # [theta_h, theta_d, 3]
        # visualising in 3d
#             figs.append(go.Figure(data=[
# #                                         go.Surface(z=brdf_slice[...,0], name='R'),
#                                         go.Surface(z=brdf_slice[...,1], name='G'),
# #                                         go.Surface(z=brdf_slice[...,2], name='B'),
#                                        ], layout=go.Layout(dict(title='iteration {:02}'.format(i+1)))))
        # visualising in 2d slice
        brdf_slice = np.swapaxes(brdf_slice, 0,1)
        fig = px.imshow(brdf_slice, zmin=-6, zmax=4, origin='lower', range_color=[-6,4], width=400, height=400, title='ground truth')
        fig.update_traces(hovertemplate="h: %{x}<br>d: %{y}<br>R:%{meta[0]:.2f}<br>G:%{meta[1]:.2f}<br>B:%{meta[2]:.2f}", meta=brdf_slice)
        figs.append(fig)

    elif case_str == '1d':
        brdf_func = load_brdf(brdf_str, log=True)
        x = np.arange(90)
        y = brdf_func(np.cos(x/180*np.pi))
        fig = go.Figure(data=[go.Scatter(x=x, y=y[...,0],name='R'),go.Scatter(x=x, y=y[...,1],name='G'),go.Scatter(x=x, y=y[...,2],name='B')], layout=go.Layout(dict(title='iteration {:02}'.format(i))) )
        figs.append(fig)
    
    
    widget_s = []
    widget = []
    for i,f in enumerate(figs):
        widget.append(go.FigureWidget(f))
        if i % 3 == 2 or i==len(figs)-1:
            widget_s.append(widgets.HBox(widget))
            widget = []
    return widgets.VBox(widget_s)


def show_views(model_str, case_str, no_views):
    mesh = trimesh.load_mesh('models/{}.obj'.format(model_str))
    v_map, l_map = np.load('results/{}-observed_map.npy'.format(model_str+'-'+case_str))
    v_map = v_map.sum(0)
    l_map = l_map.sum(0).astype(np.float32)
    lv_map = np.zeros(len(mesh.vertices))
    lv_map2 = np.zeros(len(mesh.vertices))
    vv_map = np.zeros(len(mesh.vertices))
    for (i,j,k),l,v in zip(mesh.faces, l_map, v_map):
        lv_map[i] = lv_map[i] + l / 6
        lv_map[j] = lv_map[j] + l / 6
        lv_map[k] = lv_map[k] + l / 6
        lv_map2[i] = max(lv_map2[i] , l)
        lv_map2[j] = max(lv_map2[j] , l)
        lv_map2[k] = max(lv_map2[k] , l)
        vv_map[i] = vv_map[i] + v / 6
        vv_map[j] = vv_map[j] + v / 6
        vv_map[k] = vv_map[k] + v / 6

    fig_l = go.Figure(data=[
        go.Mesh3d(
            # 8 vertices of a cube
            x=mesh.vertices[...,0],
            y=mesh.vertices[...,1],
            z=mesh.vertices[...,2],
            colorbar_title='z',
            colorscale=[[0, 'blue'],
                        [0.5, 'yellow'],
                        [1.0, 'red']],
            # Intensity of each vertex, which will be interpolated and color-coded
            intensity = np.clip(10.0 * lv_map / no_views, 0, 10),
            # i, j and k give the vertices of triangles
            i = mesh.faces[...,0],
            j = mesh.faces[...,1],
            k = mesh.faces[...,2],
            name='y',
            showscale=True,cmax=10, cmin=0,)
    ]
        ,layout=go.Layout(title = 'No. imgs. where surface is vis. and lit'))
    
    fig_v = go.Figure(data=[
        go.Mesh3d(
            # 8 vertices of a cube
            x=mesh.vertices[...,0],
            y=mesh.vertices[...,1],
            z=mesh.vertices[...,2],
            colorbar_title='z',
            colorscale=[[0, 'blue'],
                        [0.5, 'yellow'],
                        [1.0, 'red']],
            # Intensity of each vertex, which will be interpolated and color-coded
            intensity = np.clip(10.0 * vv_map / no_views, 0, 10),
            # i, j and k give the vertices of triangles
            i = mesh.faces[...,0],
            j = mesh.faces[...,1],
            k = mesh.faces[...,2],
            name='y',
            showscale=True,cmax=10, cmin=0,)
    ],layout=go.Layout(title = 'No. imgs. where surface is visible'))
    
    fig_l = go.FigureWidget(fig_l)
    fig_v = go.FigureWidget(fig_v)
    
    CAN_LOCK = [True]
    def cam_change_l(layout, camera):
        if CAN_LOCK[0]:
            CAN_LOCK[0] = False
            fig_v.layout.scene.camera = camera
            CAN_LOCK[0] = True
    def cam_change_v(layout, camera):
        if CAN_LOCK[0]:
            CAN_LOCK[0] = False
            fig_l.layout.scene.camera = camera
            CAN_LOCK[0] = True
    
    fig_l.layout.scene.on_change(cam_change_l, 'camera')
    fig_v.layout.scene.on_change(cam_change_v, 'camera')

    return widgets.HBox([fig_v, fig_l])
    
from IPython.core.display import HTML
from IPython.core.display import display as display_html
from IPython.display import display
import trimesh
def plot_all(brdf_str, model_str, case_str, it_no, no_views):
    display_html(HTML('<h2>Model "{}", BRDF "{}"({})</h2><br>'.format(model_str, brdf_str, case_str)))
    
    display_html(HTML('<h3>convergence curves</h3><br>'))
    c_widget = convergence(brdf_str, model_str, case_str)
    display(c_widget)
    
    display_html(HTML('<h3>input images</h3><br>'))
    inputs(brdf_str, model_str, case_str, no_views=30, sync_views=False)
    v_widget = show_views(model_str, case_str, no_views)
    display(v_widget)
    
    display_html(HTML('<h3>BRDF per iteration</h3><br>'))
    b_widget = BRDF(brdf_str, model_str, case_str, it_no)
    display(b_widget)
    
    display_html(HTML('<h3>shape per iteration</h3><br>'))
    s_widget = shape_widget_all(brdf_str, model_str, case_str, it_no)
    display(s_widget)
    
    display_html(HTML('<h3>rerendered image per iteration</h3><br>'))
    rerender(brdf_str, model_str, case_str, it_no)
