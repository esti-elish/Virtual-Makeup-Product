import { Component } from '@angular/core';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',

})
export class AppComponent {
  title = 'makeupAngular';
  
   showPreviewOne(event){
    if(event.target.files.length > 0){
      let src = URL.createObjectURL(event.target.files[0]);
      let preview = document.getElementsByTagName('img');
      preview[0].setAttribute("src",src); 
      preview[0].style.display = "block";
    } 
  }
   myImgRemoveFunctionOne() {
    document.getElementById("file-ip-1-preview").setAttribute("src","https://i.ibb.co/ZVFsg37/default.png");
  }

}
