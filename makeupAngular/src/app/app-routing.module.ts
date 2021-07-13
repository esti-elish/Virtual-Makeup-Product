import { NgModule } from '@angular/core';
import { Routes, RouterModule, Route } from '@angular/router';
import {MakeupComponent} from './makeupModule/makeup/makeup.component';
import { HomeComponent } from './home/home.component';

const routes: Route[] = [
     {path: "", redirectTo: "home", pathMatch:"full"},
     {path: "home", component: HomeComponent},
     {path:"makeup", component:MakeupComponent}
   
  
];


@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
