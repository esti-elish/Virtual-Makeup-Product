import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';

import {MakeupComponent} from './makeup/makeup.component'
import {HttpClientModule} from '@angular/common/http'
import { MakeupService } from './makeup/makeup.service';
@NgModule({
  declarations: [MakeupComponent],
  imports: [
    CommonModule,HttpClientModule
  ],
  providers:[MakeupService ],
  exports:[MakeupComponent]
})
export class MakeupModule { }
