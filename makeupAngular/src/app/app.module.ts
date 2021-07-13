import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { HttpClientModule } from '@angular/common/http';
import {MakeupModule} from './makeupModule/makeup.module';
import { HomeComponent } from './home/home.component'
@NgModule({
  declarations: [
    AppComponent,
    HomeComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,HttpClientModule,MakeupModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
