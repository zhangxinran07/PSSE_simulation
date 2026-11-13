# dynrun.py - 4 PSSe
from __future__ import with_statement   
import os
import sys#<- it must be first declaration
sys.path.append(os.path.join(os.path.dirname(__file__), r'..\package'))  # 将 PSSE_Simulation 文件夹添加到系统路径中


import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import matplotlib
matplotlib.use('TkAgg')
import psse_dynamic as ps_d  # 必须在 numpy/matplotlib 之前 import(PSSE 与 MKL DLL 冲突)
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import time
from multiprocessing import Process, Queue
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from identification import obj_con_iden as obj
from scipy import optimize as opt
import warnings
import time

np.set_printoptions(precision=3, suppress=True)  # 设置 numpy 打印选项
# 强行忽略来自 scipy 优化器的这个特定 UserWarning
warnings.filterwarnings('ignore', message='.*delta_grad == 0.0.*')


def worker(q, *args, **kwargs):
    """子进程包装器"""
    # ====== 新增：在系统层面把子进程的控制台输出彻底倒向黑洞 ======
    f_null = open(os.devnull, 'w')
    # 备份原子进程的 stdout 文件描述符（其实在子进程里不备份也行，因为运行完就死了）
    old_stdout_fileno = os.dup(sys.stdout.fileno()) 
    
    # 将系统的标准输出（fileno 1）强行重定向到 devnull
    os.dup2(f_null.fileno(), sys.stdout.fileno())
    # ==========================================================

    try:
        # 此时运行 run_psse_simulation，所有底层 Fortran 漏出来的打印都会被黑洞吞掉
        df, d_para = ps_d.run_psse_simulation(*args, **kwargs)
        q.put((df, d_para))
    except Exception as e:
        q.put(e)
    finally:
        # ====== 新增：恢复输出（好习惯，虽然子进程快死了） ======
        os.dup2(old_stdout_fileno, sys.stdout.fileno())
        f_null.close()

def one_time_simulation(kwargs):
    q = Queue()
    p = Process(target=worker, args=(q,), kwargs=kwargs)
    p.start()
    result = q.get()
    p.join()
    if isinstance(result, Exception):
        print(f"仿真发生错误: {result}")
        return None, None
    else:
        df, d_para = result
        #print(f"仿真成功完成，数据行数：{len(df)}")
        return df, d_para


# ------------------------------------------------------------
# 1. 全局参数（示例）
# ------------------------------------------------------------
params = {
    '负荷母线节点': 3,
    '参数下界': [500,90,0.1,0.2],
    '参数上界': [2000,200,1.0,1.0],
    '总仿真时间':60,
    '类噪声上限频率':2,
    '滤波器截止频率':10,
    '类噪声幅值':1000,
}

# ------------------------------------------------------------
# 3. 主窗口类
# ------------------------------------------------------------
class SimulationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("仿真控制界面")
        self.root.geometry("1200x800")

        # ---------- 状态变量 ----------
        self.results = []                # 存储所有结果 [(fig1,fig2,fig3,fig4,text), ...]
        self.current_index = -1          # 当前显示的结果索引
        self.follow_latest = True        # 是否自动跟随最新
        self.running = False             # 仿真是否运行中
        self.simulation_thread = None
        self.kwargs =  {
                'disturbance_type': 'ambient_a', 
                'study': 'IEEE39_RE',              # 这里的 study 变量会直接传递给子进程
                'total_time': params['总仿真时间'],
                'ambient_cutoff_frequency': params['类噪声上限频率'],
                'filter_cutoff_frequency': params['滤波器截止频率'],
                'disturbance_range': params['类噪声幅值'],
                'ld_con_chng_buses': [params['负荷母线节点']],
                'ld_con_chng_values': [[-1, -1, -1, -1, -1, 0.00, 1.8, 0.22, 0.08, 0.15, -1, -1, -1, -1, -1]]
            }
        self.busid=params['负荷母线节点']
        self.lb=params['参数下界']
        self.ub=params['参数上界']
        
        self.result_all=[]
        self.ori_data=pd.DataFrame()

        # ---------- 创建界面组件 ----------
        self._create_top_controls()
        self._create_middle_area()
        self._create_bottom_text()

        # 初始化显示
        self.clear_display()
        self.update_nav_buttons()

    # ---------- 布局构建 ----------
    def _create_top_controls(self):
        """顶部按钮栏 + 跳转控件"""
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill='x', padx=5, pady=5)

        # 原有按钮
        self.btn_start = ttk.Button(top_frame, text="开始仿真", command=self.start_simulation)
        self.btn_start.pack(side='left', padx=2)

        self.btn_stop = ttk.Button(top_frame, text="终止", command=self.stop_simulation, state='disabled')
        self.btn_stop.pack(side='left', padx=2)

        self.btn_params = ttk.Button(top_frame, text="参数设置", command=self.open_param_window)
        self.btn_params.pack(side='left', padx=2)

        self.btn_prev = ttk.Button(top_frame, text="上一个", command=self.show_prev)
        self.btn_prev.pack(side='left', padx=2)

        self.btn_next = ttk.Button(top_frame, text="下一个", command=self.show_next)
        self.btn_next.pack(side='left', padx=2)

        # 跳转到第一个 / 最后一个
        self.btn_first = ttk.Button(top_frame, text="<< 首个", command=self.jump_first)
        self.btn_first.pack(side='left', padx=2)

        self.btn_last = ttk.Button(top_frame, text="最新 >>", command=self.jump_last)
        self.btn_last.pack(side='left', padx=2)

        # k 跳转输入框 + 按钮
        ttk.Label(top_frame, text="k:").pack(side='left', padx=(10, 2))
        self.k_entry = ttk.Entry(top_frame, width=6)
        self.k_entry.pack(side='left', padx=2)
        self.btn_jump = ttk.Button(top_frame, text="跳转", command=self.jump_to_k)
        self.btn_jump.pack(side='left', padx=2)

        # 状态信息（简短显示在顶部右侧）
        self.status_label = ttk.Label(top_frame, text="就绪")
        self.status_label.pack(side='right', padx=10)

    def _create_middle_area(self):
        """中间区域：左侧图片网格 + 右侧状态信息框"""
        middle = ttk.Frame(self.root)
        middle.pack(fill='both', expand=True, padx=5, pady=5)

        # 左侧图片区域 (2x2)
        left_frame = ttk.Frame(middle)
        left_frame.pack(side='left', fill='both', expand=True)

        self.image_frames = []
        for i in range(4):
            row, col = divmod(i, 2)
            subframe = ttk.Frame(left_frame, relief='sunken', borderwidth=2)
            subframe.grid(row=row, column=col, sticky='nsew', padx=3, pady=3)
            label = ttk.Label(subframe, text=f"图片 {i+1} 区域", anchor='center')
            label.pack(fill='both', expand=True)
            self.image_frames.append(subframe)
            # 保存画布引用
            setattr(self, f'canvas_{i}', None)

        left_frame.grid_rowconfigure(0, weight=1)
        left_frame.grid_rowconfigure(1, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)
        left_frame.grid_columnconfigure(1, weight=1)

        # 右侧状态信息框 (使用 ScrolledText 可以显示多行)
        right_frame = ttk.Frame(middle, width=300)
        right_frame.pack(side='right', fill='y', padx=5)  # 只垂直填充，水平保持固定
        right_frame.pack_propagate(False)                 # 禁止根据内容调整宽度

        ttk.Label(right_frame, text="状态信息", font=('Arial', 10, 'bold')).pack(pady=2)
        self.state_text = scrolledtext.ScrolledText(right_frame, height=10, wrap='word', state='normal')
        self.state_text.pack(fill='both', expand=True)

    def _create_bottom_text(self):
        """底部文本框：显示当前 k 的详细结果"""
        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(fill='x', padx=5, pady=5)

        ttk.Label(bottom_frame, text="结果详情", font=('Arial', 10, 'bold')).pack(anchor='w')
        self.result_text = scrolledtext.ScrolledText(bottom_frame, height=6, wrap='word')
        self.result_text.pack(fill='both', expand=True)

    # ---------- 状态更新 ----------
    def update_state_info(self,info,clear=False):
        """更新右侧状态信息框"""
        self.state_text.config(state='normal')
        if clear:
            self.state_text.delete('1.0', tk.END)
        self.state_text.insert(tk.END, info)
        self.state_text.config(state='disabled')

    # ---------- 显示相关 ----------
    def clear_display(self):
        """清空图片和结果文本框"""
        for i, frame in enumerate(self.image_frames):
            for child in frame.winfo_children():
                child.destroy()
            label = ttk.Label(frame, text=f"图片 {i+1} 区域", anchor='center')
            label.pack(fill='both', expand=True)
            setattr(self, f'canvas_{i}', None)
        self.result_text.delete('1.0', tk.END)
        self.update_state_info("", clear=True)

    def display_result(self, index):
        """
        从 self.ori_data 中提取第 index 步的数据（按行连续存储）
        每步数据点数为 N = 总时间/0.01 + 1
        """
        if index < 0 or index*4+4 > len(self.ori_data.columns):
            return

        # 计算每步数据点数
        total_time = params.get('总仿真时间', 10.0)
        step = 0.01
        N = int(round(total_time / step)) + 1  # 确保整数

        # 提取该步数据（列 0,1,2,3 对应 u,P,Q,ang）
        u = self.ori_data.values[:,0+index*4]
        P = self.ori_data.values[:,1+index*4]
        Q = self.ori_data.values[:,2+index*4]
        ang = self.ori_data.values[:,3+index*4]

        # 生成时间轴（与仿真时一致）
        t = np.arange(0, total_time+step, step)

        # 创建四个独立的 Figure
        figs = []
        data_list = [u, P, Q, ang]
        titles = ['u', 'P', 'Q', 'ang']
        colors = ['r', 'g', 'b', 'm']
        for data, title, color in zip(data_list, titles, colors):
            f = Figure(figsize=(4, 3), dpi=100)
            ax = f.add_subplot(111)
            ax.plot(t, data, color + '-')
            ax.set_title(title)
            ax.set_xlabel('Time (s)')
            f.tight_layout()   # 关键：自动调整边距
            figs.append(f)

        # 更新四个图片区域
        for i in range(4):
            frame = self.image_frames[i]
            for child in frame.winfo_children():
                child.destroy()
            canvas = FigureCanvasTkAgg(figs[i], master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            setattr(self, f'canvas_{i}', canvas)

        # # 更新结果文本框
        if index < len(self.results):
            self.result_text.delete('1.0', tk.END)
            self.result_text.insert(tk.END, self.results[index])

        # 更新当前索引和导航按钮
        self.current_index = index
        self.update_nav_buttons()

    def update_nav_buttons(self):
        """根据当前索引启用/禁用导航按钮"""
        total = len(self.results)
        if total == 0:
            self.btn_prev.config(state='disabled')
            self.btn_next.config(state='disabled')
            self.btn_first.config(state='disabled')
            self.btn_last.config(state='disabled')
            self.btn_jump.config(state='disabled')
            return
        self.btn_prev.config(state='normal' if self.current_index > 0 else 'disabled')
        self.btn_next.config(state='normal' if self.current_index < total - 1 else 'disabled')
        self.btn_first.config(state='normal' if total > 0 else 'disabled')
        self.btn_last.config(state='normal' if total > 0 else 'disabled')
        self.btn_jump.config(state='normal' if total > 0 else 'disabled')

    # ---------- 导航操作 ----------
    def show_prev(self):
        if len(self.results) == 0 or self.current_index <= 0:
            return
        self.follow_latest = False
        new_idx = self.current_index - 1
        self.display_result(new_idx)
        self.status_label.config(text=f"手动浏览 (k={new_idx})")

    def show_next(self):
        if len(self.results) == 0 or self.current_index >= len(self.results) - 1:
            return
        new_idx = self.current_index + 1
        self.display_result(new_idx)
        if new_idx == len(self.results) - 1:
            self.follow_latest = True
            self.status_label.config(text="跟随最新")
        else:
            self.follow_latest = False
            self.status_label.config(text=f"手动浏览 (k={new_idx})")

    def jump_first(self):
        if len(self.results) == 0:
            return
        self.follow_latest = False
        self.display_result(0)
        self.status_label.config(text="跳转到首个 (k=0)")

    def jump_last(self):
        if len(self.results) == 0:
            return
        self.follow_latest = True
        self.display_result(len(self.results) - 1)
        self.status_label.config(text="跳转到最新")

    def jump_to_k(self):
        """从输入框读取 k 值并跳转"""
        if len(self.results) == 0:
            messagebox.showinfo("提示", "尚无结果，请先开始仿真")
            return
        try:
            k = int(self.k_entry.get())
        except ValueError:
            messagebox.showerror("错误", "请输入有效的整数 k")
            return
        if k < 0 or k >= len(self.results):
            messagebox.showerror("错误", f"k 必须在 0 ~ {len(self.results)-1} 之间")
            return
        self.follow_latest = False
        self.display_result(k)
        self.status_label.config(text=f"跳转到 k={k}")

    # ---------- 仿真控制 ----------
    def start_simulation(self):
        if self.running:
            return
        # 重置
        self.results = []
        self.current_index = -1
        self.follow_latest = True
        self.clear_display()
        self.status_label.config(text="仿真运行中...")
        self.btn_start.config(state='disabled')
        self.btn_stop.config(state='normal')
        self.running = True
        self.kwargs =  {
                'disturbance_type': 'ambient_a', 
                'study': 'IEEE39_RE',              # 这里的 study 变量会直接传递给子进程
                'total_time': params['总仿真时间'],
                'ambient_cutoff_frequency': params['类噪声上限频率'],
                'filter_cutoff_frequency': params['滤波器截止频率'],
                'disturbance_range': params['类噪声幅值'],
                'ld_con_chng_buses': [params['负荷母线节点']],
                'ld_con_chng_values': [[-1, -1, -1, -1, -1, 0.00, 1.8, 0.22, 0.08, 0.15, -1, -1, -1, -1, -1]]
            }
        self.busid=params['负荷母线节点']
        self.lb=params['参数下界']
        self.ub=params['参数上界']

        self.simulation_thread = threading.Thread(target=self.simulation_loop, daemon=True)
        self.simulation_thread.start()

    def simulation_loop(self):
        """后台线程：持续生成结果"""
        k = 0
        max_k = 1000  # 可调整
        while self.running and k <= max_k:
            t=time.localtime()
            self.update_state_info(f"\n第 {k+1} 次仿真开始...{time.strftime('%H:%M:%S', t)}")
            try:
                while True:
                    df, d_para = one_time_simulation(self.kwargs)
                    if df is not None:
                        break
                    
                u=df.values[:,[i for i in range(len(df.columns)) if "VOLT %d " % self.busid in df.columns[i]]].reshape(len(df))
                P=df.values[:,[i for i in range(len(df.columns)) if "PLOD %d[" % self.busid in df.columns[i]]].reshape(len(df))
                Q=df.values[:,[i for i in range(len(df.columns)) if "QLOD %d[" % self.busid in df.columns[i]]].reshape(len(df))
                ang=df.values[:,[i for i in range(len(df.columns)) if "ANGL %d " % self.busid in df.columns[i]]].reshape(len(df))
                self.ori_data=pd.concat([self.ori_data,pd.DataFrame(np.transpose([u,P,Q,ang]))],sort=False)

                con=lambda x: x[0]*u[0]**2-2*x[1]*x[3]*P[0]
                result_temp=[]
                fun_temp=[]
                ori_data_temp=[]
                
                for i in range(2):

                    start=time.time()
                    while True:
                        x0=np.random.uniform(self.lb,self.ub)
                        if obj(x0,df,self.busid,0.01)<1 and con(x0)>0:
                            break
                    #print(x0)
                    result=opt.minimize(obj,x0, args=(df,self.busid,0.01), method='trust-constr', bounds=np.transpose(np.array([self.lb,self.ub])), options={'disp': False},tol=1e-10)
                    result_temp.append(result.x)
                    fun_temp.append(result.fun)
                    self.update_state_info(f"\n辨识结果为{np.array2string(result.x,precision=2,separator=',',suppress_small=True)} 目标函数值%.2e 计算时间%.1f秒" % (result.fun, time.time() - start))
                        
                min_index=np.argmin(fun_temp)                
                self.results.append(result_temp[min_index])                                                  
                                    
            except Exception as e:
                self.root.after(0, lambda: self.status_label.config(text=f"仿真出错: {e}"))
                self.running = False
                break

            # 主线程更新
            self.root.after(0, self.on_new_result)
            k += 1
            time.sleep(0.1)  # 模拟计算耗时

        # 结束
        self.root.after(0, self.simulation_finished)

    def on_new_result(self):
        """新结果生成后的回调（主线程）"""
        if self.follow_latest:
            # 自动显示最新
            self.display_result(len(self.results) - 1)
            self.status_label.config(text=f"跟随最新 (k={self.current_index})")
        else:
            # 手动浏览模式：仅更新状态，不改变显示
            self.status_label.config(text=f"新结果 k={len(self.results)-1} 已生成 (当前手动浏览)")
        self.update_nav_buttons()

    def simulation_finished(self):
        self.running = False
        self.btn_start.config(state='normal')
        self.btn_stop.config(state='disabled')
        self.status_label.config(text=f"仿真结束，共 {len(self.results)} 个结果")
        if len(self.results) > 0 and self.follow_latest:
            self.display_result(len(self.results) - 1)
        self.update_state_info("仿真已完成。", clear=False)

    def stop_simulation(self):
        self.running = False
        self.btn_stop.config(state='disabled')
        self.status_label.config(text="正在终止...")

    # ---------- 参数设置窗口 ----------
    def open_param_window(self):
        if hasattr(self, 'param_window') and self.param_window is not None and self.param_window.winfo_exists():
            self.param_window.lift()
            return

        win = tk.Toplevel(self.root)
        win.title("参数设置")
        win.geometry("300x300")
        self.param_window = win

        temp_params = params.copy()
        entries = {}
        row = 0
        for key, value in params.items():
            label = ttk.Label(win, text=key + ":")
            label.grid(row=row, column=0, padx=5, pady=5, sticky='e')
            entry = ttk.Entry(win)
            entry.insert(0, str(value))
            entry.grid(row=row, column=1, padx=5, pady=5, sticky='w')
            entries[key] = entry
            row += 1

        btn_frame = ttk.Frame(win)
        btn_frame.grid(row=row, column=0, columnspan=2, pady=10)

        def save_params():
            try:
                for key, entry in entries.items():
                    val = float(entry.get())
                    params[key] = val
                messagebox.showinfo("成功", "参数已保存")
                win.destroy()
            except ValueError:
                messagebox.showerror("错误", "请输入有效的数值")

        def cancel_params():
            win.destroy()

        def restore_params():
            for key, entry in entries.items():
                entry.delete(0, tk.END)
                entry.insert(0, str(temp_params[key]))

        ttk.Button(btn_frame, text="保存", command=save_params).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="取消", command=cancel_params).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="还原", command=restore_params).pack(side='left', padx=5)

        def on_close():
            self.param_window = None
            win.destroy()
        win.protocol("WM_DELETE_WINDOW", on_close)


# ------------------------------------------------------------
# 2. 仿真计算函数（用户可替换为自己的逻辑）
# ------------------------------------------------------------

# ------------------------------------------------------------
# 4. 启动程序
# ------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = SimulationGUI(root)
    root.mainloop()
