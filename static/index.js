var submit = document.getElementById('submitButton');
submit.onclick = showImage;     

function showImage() {
    var newImage = document.getElementById('image-show').lastElementChild;
    newImage.style.visibility = "visible";
    document.getElementById('fileName').textContent = null;  
}


function loadFile(input) {
    var file = input.files[0];
    var name = document.getElementById('fileName');
    name.textContent = file.name;
    var newImage = document.createElement("img");
    newImage.setAttribute("class", 'img');
    newImage.src = URL.createObjectURL(file);   
    newImage.style.width = "70%";
    newImage.style.height = "70%";
    newImage.style.visibility = "hidden";  
    newImage.style.objectFit = "contain";
    var container = document.getElementById('image-show');
    container.appendChild(newImage);
};