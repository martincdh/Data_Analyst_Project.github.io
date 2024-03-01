const header = document.querySelector("[data-header]");

window.addEventListener("scroll", function () {

  if (window.scrollY > 1) {
    header.classList.add("active");
  } else {
    header.classList.remove("active");
  }

});


const themeToggleBtn = document.querySelector("[data-theme-btn]");

themeToggleBtn.addEventListener("click", function () {
  
    elemToggleFunc(themeToggleBtn);
    
    if (themeToggleBtn.classList.contains("active")) 
        {
            document.body.classList.remove("dark_theme");
            document.body.classList.add("light_theme");
        
            localStorage.setItem("theme", "light_theme");
        } 
        else 
        {
            document.body.classList.add("dark_theme");
            document.body.classList.remove("light_theme");
        
            localStorage.setItem("theme", "dark_theme");
    
        }
  
  });


var checkboxes = document.querySelectorAll('input[class=checkbox]'),
    checkboxArray = Array.from( checkboxes );

function confirmCheck() {
  if (this.checked) {
    document.body.classList.remove("dark_theme");
    document.body.classList.add("light_theme");    
    localStorage.setItem("theme", "light_theme");
  }
  else 
    {
        document.body.classList.add("dark_theme");
        document.body.classList.remove("light_theme");    
        localStorage.setItem("theme", "dark_theme");
    }
}

checkboxArray.forEach(function(checkbox) {
  checkbox.addEventListener('change', confirmCheck);
});