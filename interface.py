import tkinter as tk
from tkinter import ttk
import time

from matplotlib.pyplot import margins

from movie_recommendation import get_movies

def processar_entrada(texto: str) -> list[str]:
    if len(texto) == 0:
        resultados = ["Entre algum valor numerico"]

    else:
        try:
            palavras = get_movies(int(texto))
            resultados = [
                f"#{i+1} — '{p}'"
                for i, p in enumerate(palavras)
            ]
        except ValueError:
            resultados = ["Entre algum valor numerico"]

        if not resultados:
            resultados = ["⚠  Nenhum texto informado."]

    return resultados


# ─────────────────────────────────────────────
#  INTERFACE
# ─────────────────────────────────────────────
class App(tk.Tk):
    #Cores:
    BG = "#1c1c22"  # fundo geral
    SURFACE = "#2a2a33"  # fundo do input e listbox
    BORDER = "#FFFFFF"  # separadores
    ACCENT = "#E0E0E0"  # botão e barra lateral
    ACCENT_H = "#aaaaee"  # hover do botão
    TEXT = "#dddde8"  # texto principal
    MUTED = "#7777aa"  # texto secundário
    SUCCESS = "#88ccaa"  # mensagem de status

    FONT_UI  = ("Courier New", 11)
    FONT_BIG = ("Courier New", 13, "bold")
    FONT_SM  = ("Courier New", 9)

    def __init__(self):
        super().__init__()
        self.title("Sistema de Recomendação")
        self.geometry("640x520")
        self.minsize(480, 380)
        self.configure(bg=self.BG)
        self._build()

    # ── construção dos widgets ──────────────────
    def _build(self):
        # cabeçalho
        header = tk.Frame(self, bg=self.BG)
        header.pack(fill="x", padx=28, pady=(24, 0))

        tk.Label(
            header, text="Sistema de Recomendação", bg=self.BG,
            fg=self.ACCENT, font=("Courier New", 14, "bold")
        ).pack(anchor="center")

        tk.Label(
            header, text="Envie o Id do usuario e receba suas recomendações.",
            bg=self.BG, fg=self.MUTED, font=self.FONT_SM,
        ).pack(anchor="w", pady=(2, 0))

        tk.Frame(self, bg=self.BORDER, height=1).pack(
            fill="x", padx=28, pady=14
        )

        # área de input
        input_frame = tk.Frame(self, bg=self.SURFACE, bd=0)
        input_frame.pack(fill="x", padx=28)

        tk.Frame(input_frame, bg=self.ACCENT, width=3).pack(
            side="left", fill="y"
        )

        inner = tk.Frame(input_frame, bg=self.SURFACE)
        inner.pack(side="left", fill="both", expand=True, padx=12, pady=10)

        tk.Label(
            inner, text="ID:", bg=self.SURFACE,
            fg=self.MUTED, font=self.FONT_SM,
        ).pack(anchor="w")

        self.entry = tk.Entry(
            inner, bg=self.SURFACE, fg=self.TEXT,
            insertbackground=self.ACCENT,
            font=self.FONT_BIG, bd=0, relief="flat",
            highlightthickness=0,
        )
        self.entry.pack(fill="x", pady=(4, 0))
        self.entry.bind("<Return>", lambda _: self._executar())
        self.entry.focus_set()

        # botão
        btn_row = tk.Frame(self, bg=self.BG)
        btn_row.pack(fill="x", padx=28, pady=12)

        self.btn = tk.Button(
            btn_row, text="EXECUTAR  ⏎",
            bg=self.ACCENT, fg=self.BG,
            activebackground=self.ACCENT_H,
            activeforeground=self.BG,
            font=("Courier New", 10, "bold"),
            bd=0, relief="flat", cursor="hand2",
            padx=18, pady=8,
            command=self._executar,
        )
        self.btn.pack(side="right")

        self.status_lbl = tk.Label(
            btn_row, text="", bg=self.BG,
            fg=self.MUTED, font=self.FONT_SM,
        )
        self.status_lbl.pack(side="left", anchor="s")

        # separador
        tk.Frame(self, bg=self.BORDER, height=1).pack(
            fill="x", padx=28
        )

        # cabeçalho da lista
        list_header = tk.Frame(self, bg=self.BG)
        list_header.pack(fill="x", padx=28, pady=(10, 4))

        tk.Label(
            list_header, text="Recomendações: ",
            bg=self.BG, fg=self.MUTED, font=self.FONT_SM,
        ).pack(side="left")

        self.count_lbl = tk.Label(
            list_header, text="",
            bg=self.BG, fg=self.ACCENT, font=self.FONT_SM,
        )
        self.count_lbl.pack(side="right")

        # listbox + scrollbar
        list_frame = tk.Frame(self, bg=self.SURFACE)
        list_frame.pack(fill="both", expand=True, padx=28, pady=(0, 24))

        scrollbar = tk.Scrollbar(list_frame, bg=self.BORDER, troughcolor=self.SURFACE)
        scrollbar.pack(side="right", fill="y")

        self.listbox = tk.Listbox(
            list_frame,
            bg=self.SURFACE, fg=self.TEXT,
            selectbackground=self.ACCENT,
            selectforeground=self.BG,
            font=self.FONT_UI,
            bd=0, relief="flat",
            highlightthickness=0,
            activestyle="none",
            yscrollcommand=scrollbar.set,
        )
        self.listbox.pack(fill="both", expand=True, padx=2, pady=2)
        scrollbar.config(command=self.listbox.yview)

        # hover sutil na listbox
        self.listbox.bind("<Motion>", self._on_hover)

    # ── lógica ──────────────────────────────────
    def _executar(self):
        texto = self.entry.get()
        self.btn.config(state="disabled", text="  ...  ")
        self.update_idletasks()

        t0 = time.perf_counter()
        itens = processar_entrada(texto)
        elapsed = (time.perf_counter() - t0) * 1000

        self._popular_lista(itens)

        self.btn.config(state="normal", text="Procurar  ⏎")

    def _popular_lista(self, itens: list[str]):
        self.listbox.delete(0, "end")
        for item in itens:
            self.listbox.insert("end", f"  {item}")

    def _on_hover(self, event):
        idx = self.listbox.nearest(event.y)
        if idx < 0 or self.listbox.size() == 0:  # ← guard
            return



# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()