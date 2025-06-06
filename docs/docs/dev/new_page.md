# 📄 Adding a New Page

If you want to add a completely new feature to the app — for example, **continuous learning of models** — the easiest and cleanest way is to create a new, self-contained page.

Since pages in this app **do not directly communicate** with one another (except through `dcc.Store` components), isolating your feature into its own page helps avoid unnecessary complexity or deep code digging.

---

## ➕ 5 Steps to Add a New Feature Page

1. **Create the page file**  
   Add a new file named `new_page.py` in the `pages/` directory.

2. **Register the page**
   Inside the new file, include the following line:
   ```python
   dash.register_page(__name__, name = "New page", path="/path/to/the/new/page")
   ```
3. **Define the layout**  

   Create a variable or function named layout that returns the page’s content:
   ```python
   layout = html.Div([
        # Your page components here
    ])
   ```

4. **Define the callbacks** 

   Write callback functions that dynamically update your page components.
   Use the following pattern to register callbacks:
   Call this registration function inside `new_page.py`.  
   ```python
   # callbacks/your_feature.py
   def register_new_callbacks(**your_args):
      @callback(
         Output(component1, 'property1'),
         Output(component2, 'property2'),
         Input(component3, 'property3'),
         State(component4, 'property4')
      )
      def _new_callbacks(input_value, state_value):
         # Your logic here
         return updated_value1, updated_value2
   ```
    
5. **Update run.py (main app file)**  

   The app is already initialized with multi-page support:
   ```python
   app = Dash(__name__, use_pages=True)
   ```
   Ensure the new page is included using dash.page_container.

   You can also control the page's order or placement in the DropdownMenu component from here.

---


## 📚 Resources
> 💡 Best Practice: Group related callbacks into separate files inside the `callbacks/` directory. Structure them by feature or component (e.g., `graph.py`, `prediction.py`), not strictly by page.

To better understand how multi-page apps work in Dash, see the official documentation:

🔗 [https://dash.plotly.com/urls](https://dash.plotly.com/urls)