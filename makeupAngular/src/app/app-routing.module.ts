import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import {MakeupComponent} from './makeupModule/makeup/makeup.component';

const routes: Routes = [
    // {path: "", redirectTo: "makeup", pathMatch:"full"},
    {path:"makeup", component:MakeupComponent}
   
  
];


@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
