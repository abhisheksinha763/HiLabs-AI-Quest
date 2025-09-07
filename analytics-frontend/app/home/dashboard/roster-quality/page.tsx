'use client'

import { useState, useEffect } from 'react'

interface DemographicsData {
  state_city_distribution: Array<{
    [key: string]: number | { [city: string]: number }
    cities: { [city: string]: number }
  }>
  compliance_status: {
    total_providers: number
    active_licenses: number
    inactive_licenses: number
    suspended_licenses: number
    expired_licenses: number
    active_percentage: number
    inactive_percentage: number
    suspended_percentage: number
    expired_percentage: number
  }
}

export default function RosterQualityPage() {
  const [selectedState, setSelectedState] = useState('All States')
  const [demographicsData, setDemographicsData] = useState<DemographicsData | null>(null)
  const [loading, setLoading] = useState(false)
  const [availableStates, setAvailableStates] = useState<string[]>([])
  
  const usStates = [
    'All States',
    'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware',
    'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky',
    'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi',
    'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico',
    'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania',
    'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont',
    'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming'
  ]
  
  // Fetch demographics data from backend
  const fetchDemographicsData = async () => {
    // Check if data is already cached in sessionStorage
    const cachedData = sessionStorage.getItem('demographicsData')
    if (cachedData) {
      const parsed = JSON.parse(cachedData)
      setDemographicsData(parsed)
      extractAvailableStates(parsed)
      return
    }

    setLoading(true)
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001'
      const response = await fetch(`${apiUrl}/demographics`)
      const result = await response.json()
      
      if (result.status === 'success') {
        setDemographicsData(result.data)
        sessionStorage.setItem('demographicsData', JSON.stringify(result.data))
        extractAvailableStates(result.data)
      }
    } catch (error) {
      console.error('Error fetching demographics data:', error)
    } finally {
      setLoading(false)
    }
  }

  const extractAvailableStates = (data: DemographicsData) => {
    const states = data.state_city_distribution.map(item => {
      const stateKey = Object.keys(item).find(key => key !== 'cities')
      return stateKey || ''
    }).filter(state => state !== '')
    setAvailableStates(states)
  }

  useEffect(() => {
    fetchDemographicsData()
  }, [])
  
  // Mock data for charts (fallback)
  // License status data from backend or fallback
  const getLicenseStatusData = () => {
    if (demographicsData) {
      const { compliance_status } = demographicsData
      return [
        { 
          label: 'Active', 
          value: compliance_status.active_percentage, 
          count: compliance_status.active_licenses,
          color: '#22C55E' 
        },
        { 
          label: 'Inactive', 
          value: compliance_status.inactive_percentage, 
          count: compliance_status.inactive_licenses,
          color: '#06B6D4' 
        },
        { 
          label: 'Expired', 
          value: compliance_status.expired_percentage, 
          count: compliance_status.expired_licenses,
          color: '#F59E0B' 
        },
        { 
          label: 'Suspended', 
          value: compliance_status.suspended_percentage, 
          count: compliance_status.suspended_licenses,
          color: '#EF4444' 
        }
      ]
    }
    // Fallback data
    return [
      { label: 'Active', value: 45, count: 45, color: '#22C55E' },
      { label: 'Inactive', value: 28, count: 28, color: '#06B6D4' },
      { label: 'Expired', value: 18, count: 18, color: '#F59E0B' },
      { label: 'Suspended', value: 12, count: 12, color: '#EF4444' }
    ]
  }

  const missingDocumentsData = [
    { label: 'DEA', value: 3, color: '#10B981' },
    { label: 'BLS', value: 8, color: '#F59E0B' },
    { label: 'ACLS', value: 6, color: '#8B5CF6' },
    { label: 'PALS', value: 7, color: '#EF4444' },
    { label: 'LICENSE', value: 4, color: '#F97316' },
    { label: 'COI', value: 5, color: '#3B82F6' }
  ]

  // State wise provider data from backend or fallback
  const getStateWiseData = () => {
    if (demographicsData) {
      return demographicsData.state_city_distribution.map(item => {
        const stateKey = Object.keys(item).find(key => key !== 'cities')
        const stateName = stateKey || ''
        const count = typeof item[stateKey!] === 'number' ? item[stateKey!] as number : 0
        return { state: stateName, count }
      })
    }
    // Fallback data
    return [
      { state: 'California', count: 45 },
      { state: 'New York', count: 32 },
      { state: 'Texas', count: 28 },
      { state: 'Florida', count: 38 },
      { state: 'Illinois', count: 22 },
      { state: 'Pennsylvania', count: 35 }
    ]
  }

  // City wise data for selected state
  const getCityWiseData = () => {
    if (demographicsData && selectedState !== 'All States') {
      const stateData = demographicsData.state_city_distribution.find(item => 
        Object.keys(item).includes(selectedState)
      )
      if (stateData && stateData.cities) {
        return Object.entries(stateData.cities).map(([city, count]) => ({
          city,
          count
        }))
      }
    }
    return []
  }

  const BarChart = ({ data, title, color = '#3B82F6', showDropdown = false }: { data: any[], title: string, color?: string, showDropdown?: boolean }) => {
    const maxValue = Math.max(...data.map(d => d.count))
    
    return (
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="flex justify-between items-center mb-6">
          <h3 className="text-lg font-semibold text-gray-800">{title}</h3>
          {showDropdown && (
            <select
              value={selectedState}
              onChange={(e) => setSelectedState(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="All States">All States</option>
              {availableStates.map((state) => (
                <option key={state} value={state}>
                  {state}
                </option>
              ))}
            </select>
          )}
        </div>
        <div className="space-y-3">
          {data.map((item, index) => (
            <div key={index} className="flex items-center">
              <div className="w-16 text-sm text-gray-600 font-medium">
                {item.month || item.specialty || item.state || item.city}
              </div>
              <div className="flex-1 mx-3">
                <div className="bg-gray-200 rounded-full h-6 relative">
                  <div 
                    className="h-6 rounded-full flex items-center justify-end pr-2"
                    style={{ 
                      width: `${(item.count / maxValue) * 100}%`,
                      backgroundColor: color
                    }}
                  >
                    <span className="text-white text-xs font-medium">{item.count}</span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    )
  }

  const PieChart = ({ data, title }: { data: any[], title: string }) => {
    const total = data.reduce((sum, item) => sum + item.value, 0)
    let cumulativePercentage = 0
    const [hoveredSegment, setHoveredSegment] = useState<number | null>(null)

    return (
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-6">{title}</h3>
        <div className="flex items-center justify-center">
          <div className="relative w-64 h-64">
            <svg className="w-64 h-64 transform -rotate-90" viewBox="0 0 100 100">
              {data.map((item, index) => {
                const percentage = (item.value / total) * 100
                const strokeDasharray = `${percentage} ${100 - percentage}`
                const strokeDashoffset = -cumulativePercentage
                cumulativePercentage += percentage
                
                return (
                  <circle
                    key={index}
                    cx="50"
                    cy="50"
                    r="31.83"
                    fill="transparent"
                    stroke={item.color}
                    strokeWidth="8"
                    strokeDasharray={strokeDasharray}
                    strokeDashoffset={strokeDashoffset}
                    className="transition-all duration-300 cursor-pointer"
                    onMouseEnter={() => setHoveredSegment(index)}
                    onMouseLeave={() => setHoveredSegment(null)}
                  />
                )
              })}
            </svg>
            {/* Tooltip */}
            {hoveredSegment !== null && (
              <div className="absolute top-0 left-0 bg-gray-800 text-white px-2 py-1 rounded text-xs z-10">
                {data[hoveredSegment].label}: {data[hoveredSegment].count || Math.round((data[hoveredSegment].value / total) * 100)}
              </div>
            )}
          </div>
          <div className="ml-12 space-y-2">
            {data.map((item, index) => (
              <div key={index} className="flex items-center">
                <div 
                  className="w-4 h-4 rounded-full mr-3"
                  style={{ backgroundColor: item.color }}
                />
                <span className="text-sm font-medium text-gray-700">{item.label}</span>
                <span className="ml-2 text-sm text-gray-500">
                  {Math.round((item.value / total) * 100)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Top - State Wise Provider Distribution Bar Chart (Full Width) */}
      <div className="w-full">
        <BarChart 
          data={getStateWiseData()} 
          title="State Wise Provider Distribution" 
          color="#10B981"
        />
      </div>
      
      {/* Bottom Row - Two columns */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Bottom Left - License Status Pie Chart */}
        <PieChart 
          data={getLicenseStatusData()} 
          title="License Status"
        />
        
        {/* Bottom Right - City Wise Provider Distribution Bar Chart */}
        <BarChart 
          data={getCityWiseData()} 
          title="City Wise Provider Distribution" 
          color="#F59E0B"
          showDropdown={true}
        />
      </div>
    </div>
  )
}
