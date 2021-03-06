! PURPOSE:
!	  Writes the ThAlert.htm file. The data is composed in a table and
!	  color-coded to indicate alert levels.

	subroutine alerthtm(numStations,outputFolder,ulog,unitNumber,&
 	 datimb,stationNumber,deficit,intensity,avgIntensity,runningIntens,&
 	 antecedRainfall,in2mm,duration,stationLocation,TavgIntensity,sAWI,&
         AWIThresh,fieldCap,checkS,checkA)
 	 
! FORMAL ARGUMENTS
	character, intent(in) 	       :: outputFolder*(*)
	character(*), intent(in)       :: stationNumber(numStations)
	character (len=50), intent(in) :: stationLocation(numStations)
	character (len=20), intent(in) :: datimb(numStations)
	real, intent(in) 	       :: antecedRainfall(numStations)
	real, intent(in) 	       :: duration(numStations),in2mm
	real, intent(in) 	       :: deficit(numStations),runningIntens
	real, intent(in)	       :: intensity(numStations),avgIntensity(numStations),sAWI(numStations)
	real, intent(in)               :: AWIThresh,fieldCap,TavgIntensity
	integer, intent(in) 	       :: numStations,unitNumber,ulog
        logical, intent(in)            :: checkS,checkA	
	
! LOCAL VARIABLES
	character (len=57)	     :: hexColorCenter(4)
	character (len=38)	     :: hexColor(4)
	character (len=7) 	     :: alert_lev(4) !,color(4) 
	character 		     :: tb = char(9)
	character (len=255) 	     :: outputFile='ThAlert.htm'
	character (len=8)            :: TavgIntensityF 
	character (len=31),parameter :: r1c='<tr style="text-align:center;">'
	character (len=4),parameter  :: h1='<th>',d1='<td>', r1='<tr>'
	character (len=5),parameter  :: r2='</tr>',h2='</th>',d2='</td>'
	real			     :: avgToRunning,AWI_low
	integer 		     :: i,alertConditionRecentAntecedent(numStations)
	integer 		     :: alertConditionID(numStations)
	integer 		     :: alertConditionIA(numStations)
	
!------------------------------	
! alert levels 0="Null", 1="Outlook", 2="Watch", 3="Warning"	
! Color names corresponding to hexColor: color=(/'Grey',' Yellow ','Orange ','  Red  '/)
       alert_lev=(/'Null   ','Outlook','Watch  ','Warning'/)
       hexColorCenter=(/'<td style="text-align:center; background-color:#cccccc;">',&
       '<td style="text-align:center; background-color:#ffff33;">',&
       '<td style="text-align:center; background-color:#ff6600;">',&
       '<td style="text-align:center; background-color:#ff0000;">'/)
       hexColor=(/'<td style="background-color:#cccccc;">',&
       '<td style="background-color:#ffff33;">',&
       '<td style="background-color:#ff6600;">',&
       '<td style="background-color:#ff0000;">'/)

! determine alert condition for Cumulative Recent & Antecedent threshold
	do i=1,numStations	
	  alertConditionRecentAntecedent(i)=0
	  if(deficit(i) >-0.5 .and. deficit(i) <0.d0) then
	    alertConditionRecentAntecedent(i)=0
	  else if (deficit(i) >=0.) then
	    alertConditionRecentAntecedent(i)=1
	  end if
	end do
	
        if(checkA .and. .not. checkS) then
! determine alert condition for Intensity-Duration Threshold & AWI
           AWI_low=-(AWIThresh+fieldCap)/2.
           do i = 1, numStations
        	     alertConditionID(i)=0
        	     if(duration(i)>0 .and. deficit(i) >=0.) then  ! threshold applicable only if duration>0
        	       alertConditionID(i)=1
        	       if(intensity(i) >1.0 .and. sAWI(i) > AWI_low) then
        	          alertConditionID(i)=2
        	       else if(intensity(i) >=1.0 .and. sAWI(i) > AWIThresh) then
        	         alertConditionID(i)=3
        	       end if
        	     end if
        	   end do
	else	
! determine alert condition for Intensity-Duration Threshold alone
	   do i = 1, numStations
	     alertConditionID(i)=0
	     if(duration(i)>0 .and. deficit(i) >=0.) then  ! threshold applicable only if duration>0
	       alertConditionID(i)=1
	       if(intensity(i) > 0.9 .and. intensity(i) < 1.0 ) then
	         alertConditionID(i)=2
	       else if(intensity(i) >= 1.0) then
	         alertConditionID(i)=3
	       end if
	     end if
	   end do
	end if
		
! determine alert condition for Intensity-Antecedent Precipitation Threshold
	do i = 1, numStations
	  alertConditionIA(i)=0
	  if(deficit(i) > 0) then ! antecedent
	      alertConditionIA(i)=1
	      avgToRunning = avgIntensity(i) / runningIntens
	    if(avgToRunning >0.9 .and. avgToRunning < 1.0) then
	      alertConditionIA(i)=2
	    else if(avgToRunning >= 1.0) then
	      alertConditionIA(i)=3
	    end if
	  end if
	end do
	
! file in html-table format listing most recent alert conditions at all stations
        write(TavgIntensityF,'(F8.3)') TavgIntensity
        TavgIntensityF=adjustl(TavgIntensityF)
  	outputFile=trim(outputFolder)//trim(outputFile)
  	open(unitNumber,file=outputFile,status='unknown',&
     	position='rewind',err=125)
     	write (unitNumber,*) '<table style="width:90%; border: 1px solid black; margin:auto;">'
	write (unitNumber,*) '<caption>',&
	' Current Alert Levels by Station and Threshold ',&
	'</caption>'
        if(checkA .and. .not. checkS) then
	   write (unitNumber,*) r1c,h1,'Rain gage',h2,h1,'Vicinity',h2,h1,&
           'Time and Date',h2,h1,'Recent-antecedent',h2,h1,'Intensity-duration and AWI',&
           h2,h1,trim(TavgIntensityF),'-h running average intensity',h2,r2
        else
	   write (unitNumber,*) r1c,h1,'Rain gage',h2,h1,'Vicinity',h2,h1,&
           'Time and Date',h2,h1,'Recent-antecedent',h2,h1,'Intensity-duration',&
           h2,h1,trim(TavgIntensityF),'-h running average intensity',h2,r2
        end if
	do i=1,numStations
	  write(unitNumber,*) &
          r1c,d1,trim(stationNumber(i)),d2,d1,trim(stationLocation(i)),d2,&
          d1,datimb(i),d2,hexColor(1+alertConditionRecentAntecedent(i)),&
          trim(adjustl(alert_lev(1+alertConditionRecentAntecedent(i)))),d2,&
          hexColor(1+alertConditionID(i)),trim(adjustl(alert_lev(1+alertConditionID(i)))),&
          d2,hexColor(1+alertConditionIA(i)),trim(adjustl(alert_lev(1+alertConditionIA(i)))),d2,r2
	end do
	write (unitNumber,*) '</table>'
	write (unitNumber,*) '<p>'
    	write (unitNumber,*) '<table style="width:70%; margin:auto;">'
	write (unitNumber,*) '<caption>',' EXPLANATION ','</caption>'
	write (unitNumber,*) r1,hexColorCenter(1),trim(adjustl(alert_lev(1))),d2,d1,&
        '&emsp; Landslides very unlikely',d2,r2,&
        r1,hexColorCenter(2),trim(adjustl(alert_lev(2))),d2,d1,&
        '&emsp; Landslides might occur with additional rainfall',d2,r2,&
        r1,hexColorCenter(3),trim(adjustl(alert_lev(3))),d2,d1,&
        '&emsp; Landslides likely with additional rainfall',d2,r2,&
        r1,hexColorCenter(4),trim(adjustl(alert_lev(4))),d2,d1,'&emsp; Landslides very likely',d2,r2
	write (unitNumber,*) '</table>'
  	close(unitNumber)
	write(*,*) 'Finished alert status HTML page'
	return
	
! WRITES ERROR MESSAGE TO LOG FILE
  125	write(ulog,*) 'Error opening file ',trim(outputFile)		
  	close (ulog)
	stop
	end subroutine alerthtm
